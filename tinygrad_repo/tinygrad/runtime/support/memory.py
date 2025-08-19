import collections, functools, dataclasses
from typing import Any, ClassVar
from tinygrad.helpers import round_up, getenv

class TLSFAllocator:
  """
  The allocator is based on the Two-Level Segregated Fit (TLSF) algorithm. The allocator maintains 2 level of buckets:
    * 1st level is determined by the most significant bit of the size.
    * 2nd level splits the covered memory of 1st level into @lv2_cnt entries.

  For each allocation request, the allocator searches for the smallest block that can fit the requested size.
  For each deallocation request, the allocator merges the block with its neighbors if they are free.
  """

  def __init__(self, size:int, base:int=0, block_size:int=16, lv2_cnt:int=16):
    self.size, self.base, self.block_size, self.l2_cnt = size, base, block_size, lv2_cnt.bit_length()
    self.storage:list = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
    self.lv1_entries:list[int] = [0] * len(self.storage)

    # self.blocks is more like a linked list, where each entry is a contiguous block.
    self.blocks:dict[int, tuple[int, int|None, int|None, bool]] = {0: (size, None, None, True)} # size, next, prev, is_free
    self._insert_block(0, size)

  @functools.cache
  def lv1(self, size): return size.bit_length()

  @functools.cache
  def lv2(self, size): return (size - (1 << (size.bit_length() - 1))) // (1 << max(0, size.bit_length() - self.l2_cnt))

  def _insert_block(self, start:int, size:int, prev:int|None=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].append(start)
    self.lv1_entries[self.lv1(size)] += 1
    self.blocks[start] = (size, start + size, prev, True)
    return self

  def _remove_block(self, start:int, size:int, prev:int|None=None):
    if prev is None: prev = self.blocks[start][2]
    self.storage[self.lv1(size)][self.lv2(size)].remove(start)
    self.lv1_entries[self.lv1(size)] -= 1
    self.blocks[start] = (size, start + size, prev, False)
    return self

  def _split_block(self, start:int, size:int, new_size:int):
    nxt = self.blocks[start][1]
    assert self.blocks[start][3], "block must be free"
    self._remove_block(start, size)._insert_block(start, new_size)._insert_block(start + new_size, size - new_size, prev=start)
    if nxt in self.blocks: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start + new_size, self.blocks[nxt][3])
    return self

  def _merge_right(self, start:int):
    size, nxt, _, is_free = self.blocks[start]
    assert is_free, "block must be free"

    while is_free and nxt in self.blocks:
      if (blk:=self.blocks[nxt])[3] is False: break
      self._remove_block(start, size)._remove_block(nxt, blk[0])._insert_block(start, size:=size + blk[0])
      assert self.blocks[start][1] == blk[1]
      _, nxt, _, _ = self.blocks.pop(nxt)

    if nxt in self.blocks: self.blocks[nxt] = (self.blocks[nxt][0], self.blocks[nxt][1], start, self.blocks[nxt][3])

  def _merge_block(self, start:int):
    # Go left while blocks are free. Then merge all them right.
    while (x:=self.blocks[start][2]) is not None and self.blocks[x][3] is True: start = x
    self._merge_right(start)

  def alloc(self, req_size:int, align:int=1) -> int:
    req_size = max(self.block_size, req_size) # at least block size.
    size = max(self.block_size, req_size + align - 1)

    # Round up the allocation size to the next bucket, so any entry there can fit the requested size.
    size = round_up(size, (1 << size.bit_length() - self.l2_cnt))

    # Search for the smallest block that can fit the requested size. Start with the it's bucket and go up until any block is found.
    for l1 in range(self.lv1(size), len(self.storage)):
      if self.lv1_entries[l1] == 0: continue
      for l2 in range(self.lv2(size) if l1 == size.bit_length() else 0, (1 << self.l2_cnt)):
        if len(self.storage[l1][l2]) > 0:
          nsize = self.blocks[self.storage[l1][l2][0]][0]
          assert nsize >= size, "block must be larger"

          # Block start address.
          start = self.storage[l1][l2][0]

          # If request contains alignment, split the block into two parts.
          if (new_start:=round_up(start, align)) != start:
            self._split_block(start, nsize, new_start - start)
            start, nsize = new_start, self.blocks[new_start][0]

          # If the block is larger than the requested size, split it into two parts.
          if nsize > req_size: self._split_block(start, nsize, req_size)
          self._remove_block(start, req_size) # Mark the block as allocated.
          return start + self.base
    raise MemoryError(f"Can't allocate {req_size} bytes")

  def free(self, start:int):
    self._insert_block(start - self.base, self.blocks[start - self.base][0])._merge_block(start - self.base)

# Memory Managment

@dataclasses.dataclass(frozen=True)
class VirtMapping: va_addr:int; size:int; paddrs:list[tuple[int, int]]; uncached:bool=False; system:bool=False; snooped:bool=False # noqa: E702

class PageTableTraverseContext:
  def __init__(self, dev, pt, vaddr, create_pts=False, free_pts=False, boot=False):
    self.dev, self.vaddr, self.create_pts, self.free_pts, self.boot = dev, vaddr - dev.mm.va_base, create_pts, free_pts, boot
    self.pt_stack:list[tuple[Any, int, int]] = [(pt, self._pt_pte_idx(pt, self.vaddr), self._pt_pte_size(pt))]

  def _pt_pte_cnt(self, lv): return self.dev.mm.pte_cnt[lv]
  def _pt_pte_size(self, pt): return self.dev.mm.pte_covers[pt.lv]
  def _pt_pte_idx(self, pt, va): return (va // self._pt_pte_size(pt)) % self._pt_pte_cnt(pt.lv)

  def level_down(self):
    pt, pte_idx, _ = self.pt_stack[-1]

    if not pt.valid(pte_idx):
      assert self.create_pts, "Not allowed to create new page table"
      pt.set_entry(pte_idx, self.dev.mm.palloc(0x1000, zero=True, boot=self.boot), table=True, valid=True)

    assert not pt.is_huge_page(pte_idx), f"Must be table pt={pt.paddr:#x}, {pt.lv=} {pte_idx=} {pt.read_fields(pte_idx)}"
    child_page_table = self.dev.mm.pt_t(self.dev, pt.address(pte_idx), lv=pt.lv+1)

    self.pt_stack.append((child_page_table, self._pt_pte_idx(child_page_table, self.vaddr), self._pt_pte_size(child_page_table)))
    return self.pt_stack[-1]

  def _try_free_pt(self) -> bool:
    pt, _, _ = self.pt_stack[-1]
    if self.free_pts and pt != self.dev.mm.root_page_table and all(not pt.valid(i) for i in range(self._pt_pte_cnt(self.pt_stack[-1][0].lv))):
      self.dev.mm.pfree(pt.paddr)
      parent_pt, parent_pte_idx, _ = self.pt_stack[-2]
      parent_pt.set_entry(parent_pte_idx, 0x0, valid=False)
      return True
    return False

  def level_up(self):
    while self._try_free_pt() or self.pt_stack[-1][1] == self._pt_pte_cnt(self.pt_stack[-1][0].lv):
      pt, pt_cnt, _ = self.pt_stack.pop()
      if pt_cnt == self._pt_pte_cnt(pt.lv): self.pt_stack[-1] = (self.pt_stack[-1][0], self.pt_stack[-1][1] + 1, self.pt_stack[-1][2])

  def next(self, size:int, paddr:int|None=None, off:int=0):
    while size > 0:
      pt, pte_idx, pte_covers = self.pt_stack[-1]
      if self.create_pts:
        assert paddr is not None, "paddr must be provided when allocating new page tables"
        while pte_covers > size or not pt.supports_huge_page(paddr+off) or self.vaddr&(pte_covers-1) != 0: pt, pte_idx, pte_covers = self.level_down()
      else:
        while not pt.is_huge_page(pte_idx): pt, pte_idx, pte_covers = self.level_down()

      entries = min(size // pte_covers, self._pt_pte_cnt(pt.lv) - pte_idx)
      assert entries > 0, f"Invalid entries {size=:#x}, {pte_covers=:#x}"
      yield off, pt, pte_idx, entries, pte_covers

      size, off, self.vaddr = size - entries * pte_covers, off + entries * pte_covers, self.vaddr + entries * pte_covers
      self.pt_stack[-1] = (pt, pte_idx + entries, pte_covers)
      self.level_up()

class MemoryManager:
  va_allocator: ClassVar[TLSFAllocator|None] = None

  def __init__(self, dev, vram_size:int, boot_size:int, pt_t, va_bits:int, va_shifts:list[int], va_base:int,
               palloc_ranges:list[tuple[int, int]], first_lv:int=0):
    self.dev, self.vram_size, self.va_shifts, self.va_base, lvl_msb = dev, vram_size, va_shifts, va_base, va_shifts + [va_bits + 1]
    self.pte_covers, self.pte_cnt = [1 << x for x in va_shifts][::-1], [1 << (lvl_msb[i+1] - lvl_msb[i]) for i in range(len(lvl_msb) - 1)][::-1]
    self.pt_t, self.palloc_ranges, self.level_cnt, self.va_bits = pt_t, palloc_ranges, len(va_shifts), va_bits

    self.boot_allocator = TLSFAllocator(boot_size, base=0) # per device
    self.pa_allocator = TLSFAllocator(vram_size - (64 << 20), base=self.boot_allocator.size) # per device
    self.root_page_table = pt_t(self.dev, self.palloc(0x1000, zero=not self.dev.smi_dev, boot=True), lv=first_lv)

  def _frag_size(self, va, sz, must_cover=True):
    """
    Calculate the tlb fragment size for a given virtual address and size.
    If must_cover is True, the fragment size must cover the size, otherwise the biggest fragment size that fits the size is returned.
    Fragment 0 is 4KB, 1 is 8KB and so on.
    """
    va_pwr2_div, sz_pwr2_div, sz_pwr2_max = va & -(va) if va > 0 else (1 << 63), sz & -(sz), (1 << (sz.bit_length() - 1))
    return (min(va_pwr2_div, sz_pwr2_div) if must_cover else min(va_pwr2_div, sz_pwr2_max)).bit_length() - 1 - 12

  def page_tables(self, vaddr:int, size:int):
    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True)
    for _ in ctx.next(size, paddr=0): return [pt for pt, _, _ in ctx.pt_stack]

  def map_range(self, vaddr:int, size:int, paddrs:list[tuple[int, int]], uncached=False, system=False, snooped=False, boot=False) -> VirtMapping:
    if getenv("MM_DEBUG", 0): print(f"mm {self.dev.devfmt}: mapping {vaddr=:#x} ({size=:#x})")

    assert size == sum(p[1] for p in paddrs), f"Size mismatch {size=} {sum(p[1] for p in paddrs)=}"

    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, create_pts=True, boot=boot)
    for paddr, psize in paddrs:
      for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(psize, paddr=paddr):
        for pte_off in range(pte_cnt):
          assert not pt.valid(pte_idx + pte_off), f"PTE already mapped: {pt.entry(pte_idx + pte_off):#x}"
          pt.set_entry(pte_idx + pte_off, paddr + off + pte_off * pte_covers, uncached=uncached, system=system, snooped=snooped,
                       frag=self._frag_size(ctx.vaddr+off, pte_cnt * pte_covers), valid=True)

    self.on_range_mapped()
    return VirtMapping(vaddr, size, paddrs, uncached=uncached, system=system, snooped=snooped)

  def unmap_range(self, vaddr:int, size:int):
    if getenv("MM_DEBUG", 0): print(f"mm {self.dev.devfmt}: unmapping {vaddr=:#x} ({size=:#x})")

    ctx = PageTableTraverseContext(self.dev, self.root_page_table, vaddr, free_pts=True)
    for off, pt, pte_idx, pte_cnt, pte_covers in ctx.next(size):
      for pte_id in range(pte_idx, pte_idx + pte_cnt):
        assert pt.valid(pte_id), f"PTE not mapped: {pt.entry(pte_id):#x}"
        pt.set_entry(pte_id, paddr=0x0, valid=False)

  def on_range_mapped(self): pass

  @classmethod
  def alloc_vaddr(cls, size:int, align=0x1000) -> int:
    assert cls.va_allocator is not None, "must be set it"
    return cls.va_allocator.alloc(size, max((1 << (size.bit_length() - 1)), align))

  def valloc(self, size:int, align=0x1000, uncached=False, contiguous=False) -> VirtMapping:
    # Alloc physical memory and map it to the virtual address
    va = self.alloc_vaddr(size:=round_up(size, 0x1000), align)

    if contiguous: paddrs = [(self.palloc(size, zero=True), size)]
    else:
      # Traverse the PT to find the largest contiguous sizes we need to allocate. Try to allocate the longest segment to reduce TLB pressure.
      nxt_range, rem_size, paddrs = 0, size, []
      while rem_size > 0:
        while self.palloc_ranges[nxt_range][0] > rem_size: nxt_range += 1

        try: paddrs += [(self.palloc(try_sz:=self.palloc_ranges[nxt_range][0], self.palloc_ranges[nxt_range][1], zero=False), try_sz)]
        except MemoryError:
          # Move to a smaller size and try again.
          nxt_range += 1
          if nxt_range == len(self.palloc_ranges):
            for paddr, _ in paddrs: self.pa_allocator.free(paddr)
            raise MemoryError(f"Failed to allocate memory. (total allocation size={size:#x}, current try={self.palloc_ranges[nxt_range-1]})")
          continue
        rem_size -= self.palloc_ranges[nxt_range][0]

    return self.map_range(va, size, paddrs, uncached=uncached)

  def vfree(self, vm:VirtMapping):
    assert self.va_allocator is not None, "must be set it"
    self.unmap_range(vm.va_addr, vm.size)
    self.va_allocator.free(vm.va_addr)
    for paddr, _ in vm.paddrs: self.pa_allocator.free(paddr)

  def palloc(self, size:int, align:int=0x1000, zero=True, boot=False) -> int:
    assert self.dev.is_booting == boot, "During booting, only boot memory can be allocated"
    paddr = (self.boot_allocator if boot else self.pa_allocator).alloc(round_up(size, 0x1000), align)
    if zero: self.dev.vram[paddr:paddr+size] = bytes(size)
    return paddr

  def pfree(self, paddr:int): self.pa_allocator.free(paddr)
