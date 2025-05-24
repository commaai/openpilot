import collections, functools
from tinygrad.helpers import round_up

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

    # self.blocks is more like a linked list, where each entry is a contigous block.
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
