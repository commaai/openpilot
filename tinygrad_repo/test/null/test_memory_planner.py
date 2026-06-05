import unittest
from tinygrad import dtypes
from tinygrad.uop.ops import UOp, Ops
from tinygrad.schedule.memory import memory_plan_rewrite

global_map = {}
held_bufs: set[UOp] = set()
def b(i, base=None, offset=0, pin=False, size=16):
  global global_map
  if i in global_map: return global_map[i]
  if base is not None:
    global_map[i] = global_map[base]
    return global_map[i]
  global_map[i] = UOp.new_buffer("NULL", size, dtypes.int8)
  if pin: held_bufs.add(global_map[i])
  return global_map[i]

def _make_linear(buffer_lists, copies=None):
  copy_pairs = {frozenset((id(dst), id(src))) for dst, src in copies} if copies else set()
  calls = []
  for bufs in buffer_lists:
    is_copy = len(bufs) == 2 and frozenset((id(bufs[0]), id(bufs[1]))) in copy_pairs
    calls.append(UOp(Ops.CALL, dtypes.void, (UOp(Ops.COPY if is_copy else Ops.SINK), *bufs)))
  return UOp(Ops.LINEAR, src=tuple(calls))

def _get_arena(buf, linear, result):
  for orig_si, new_si in zip(linear.src, result.src):
    for orig, new in zip(orig_si.src[1:], new_si.src[1:]):
      if orig is buf and new.op is Ops.BUFFER_VIEW: return new.src[0]
  return None

def check_assign(buffer_lists, copies=None):
  linear = _make_linear(buffer_lists, copies)
  result = memory_plan_rewrite(linear, held_bufs)

  # build mapping: original buf -> (arena, offset_bytes, nbytes) from the result
  replace_map: dict[int, tuple[UOp, int, int]] = {}
  for orig_si, new_si in zip(linear.src, result.src):
    for orig, new in zip(orig_si.src[1:], new_si.src[1:]):
      if new.op is Ops.BUFFER_VIEW and id(orig) not in replace_map:
        replace_map[id(orig)] = (new.src[0], new.arg[1] * new.dtype.itemsize, new.arg[0] * new.dtype.itemsize)

  # verify pinned buffers are not planned
  for buf in held_bufs:
    assert id(buf) not in replace_map, "pinned buffer was planned"

  # compute lifetimes
  first_appearance, last_appearance = {}, {}
  for i, bufs in enumerate(buffer_lists):
    for buf in bufs:
      if buf in held_bufs: continue
      if id(buf) not in first_appearance: first_appearance[id(buf)] = i
      last_appearance[id(buf)] = i

  # verify non-overlapping: no two live buffers share the same arena region
  taken_parts: set[tuple[int, int, int, int]] = set()  # (id(arena), offset, nbytes, id(buf))
  for i, bufs in enumerate(buffer_lists):
    for buf in bufs:
      if buf in held_bufs or id(buf) not in replace_map: continue
      arena, off, nb = replace_map[id(buf)]
      for part in taken_parts:
        assert id(buf) == part[3] or part[0] != id(arena) or part[1] + part[2] <= off or part[1] >= off + nb, \
          f"overlap at step {i}: [{off}, {off+nb}) conflicts with [{part[1]}, {part[1]+part[2]})"
      if first_appearance.get(id(buf)) == i: taken_parts.add((id(arena), off, nb, id(buf)))
      if last_appearance.get(id(buf)) == i: taken_parts.discard((id(arena), off, nb, id(buf)))

class TestMemoryPlanner(unittest.TestCase):
  def setUp(self):
    global global_map
    held_bufs.clear()
    global_map = {}

  def test_simple_buffer(self):
    bs = [
      [b(0), b(1), b(2)],
      [b(1), b(2), b(3)],
      [b(4), b(3)],
      [b(5), b(2)],
    ]
    check_assign(bs)

  def test_simple_pinned(self):
    bs = [
      [b(0, pin=True), b(1), b(2, pin=True)],
      [b(1), b(2), b(3)],
      [b(4), b(3)],
      [b(5), b(2)],
    ]
    check_assign(bs)

  def test_all_pinned(self):
    bs = [
      [b(0, pin=True), b(1, pin=True)],
      [b(1), b(2, pin=True)],
      [b(4, pin=True), b(3, pin=True)],
    ]
    check_assign(bs)

  def test_simple_buffer_offset(self):
    bs = [
      [b(0, pin=True), b(1, base=0, offset=1, size=8), b(2)],
      [b(1), b(2), b(3, base=0, offset=1, size=8)],
      [b(4), b(3)],
    ]
    check_assign(bs)

  def test_buffer_offset(self):
    bs = [
      [b(0, pin=True), b(1, base=0, offset=1, size=8), b(2)],
      [b(1), b(2), b(3, base=0, offset=1, size=8)],
      [b(4), b(3)],
      [b(5, base=2, offset=2, size=8), b(3)],
      [b(6), b(5), b(0)],
      [b(7), b(8, pin=True)],
      [b(8), b(9, base=2, offset=2, size=8)],
      [b(9), b(3), b(5)],
    ]
    check_assign(bs)

  def test_buffer_offset2(self):
    bs = [
      [b(0, pin=True), b(1), b(2)],
      [b(1), b(2), b(3)],
      [b(4), b(3)],
      [b(5), b(3)],
      [b(6), b(5), b(0)],
      [b(7), b(8, pin=True)],
      [b(8), b(9)],
      [b(9), b(3), b(5)],
      [b(11), b(0)],
      [b(11), b(10), b(5)],
      [b(12), b(11), b(0)],
      [b(6), b(12), b(7)],
      [b(13), b(6), b(11)],
    ]
    check_assign(bs)

  def test_all_offsets_of_one(self):
    bs = [
      [b(0, pin=True), b(1)],
      [b(3, base=1, offset=0, size=8), b(2, base=0, offset=0, size=8)],
      [b(5, base=1, offset=8, size=8), b(4, base=0, offset=8, size=8)],
      [b(7, base=1, offset=4, size=8), b(6, base=0, offset=4, size=8)],

      [b(4), b(5), b(2)],
      [b(3), b(7)],
      [b(10), b(6), b(7)],
      [b(11), b(3), b(2)],
      [b(12), b(5), b(4), b(3), b(2)],
      [b(13), b(6), b(12), b(7)],
    ]
    check_assign(bs)

  def test_very_small_buffers(self):
    bs = [
      [b(0, pin=True), b(1, size=32)],
      [b(3, size=4), b(4, size=6)],
    ]
    check_assign(bs)

  def test_very_big_buffers(self):
    bs = [
      [b(0, pin=True), b(1, size=34359738368000)],
      [b(3, size=1 << 128), b(4, size=1 << 64)],
    ]
    check_assign(bs)

  def test_copy_bufs_separate_from_compute(self):
    bs = [
      [b(0), b(1)],
      [b(1), b(2)],
      [b(3), b(2)],
    ]
    linear = _make_linear(bs, copies=[(b(1), b(0))])
    result = memory_plan_rewrite(linear)
    r1_arena, r2_arena = _get_arena(b(1), linear, result), _get_arena(b(2), linear, result)
    assert r1_arena is not None and r2_arena is not None
    assert r1_arena is not r2_arena

  def test_copy_bufs_reuse_among_copies(self):
    bs = [
      [b(0), b(1)],
      [b(2), b(1)],
      [b(3), b(2)],
    ]
    linear = _make_linear(bs, copies=[(b(1), b(0)), (b(2), b(1))])
    result = memory_plan_rewrite(linear)
    r1_arena, r2_arena = _get_arena(b(1), linear, result), _get_arena(b(2), linear, result)
    assert r1_arena is not None and r2_arena is not None
    assert r1_arena is r2_arena

  def test_compute_bufs_reuse_among_compute(self):
    bs = [
      [b(0), b(1)],
      [b(2), b(1)],
      [b(3), b(2)],
      [b(4), b(3)],
    ]
    linear = _make_linear(bs, copies=[(b(1), b(0))])
    result = memory_plan_rewrite(linear)
    r2_arena, r3_arena = _get_arena(b(2), linear, result), _get_arena(b(3), linear, result)
    assert r2_arena is not None and r3_arena is not None
    assert r2_arena is r3_arena

  def test_copy_and_compute_no_cross_reuse(self):
    bs = [
      [b(0), b(1)],
      [b(2), b(1)],
      [b(3), b(2)],
    ]
    linear = _make_linear(bs, copies=[(b(2), b(1))])
    result = memory_plan_rewrite(linear)
    r0_arena, r2_arena = _get_arena(b(0), linear, result), _get_arena(b(2), linear, result)
    assert r0_arena is not None and r2_arena is not None
    assert r0_arena is not r2_arena

  def test_multiple_copy_bufs_with_offsets(self):
    bs = [
      [b(0, pin=True), b(1), b(2)],
      [b(3, base=0, offset=1, size=8), b(1), b(2)],
      [b(4), b(3)],
      [b(5), b(4)],
    ]
    check_assign(bs, copies=[(b(1), b(0)), (b(2), b(0))])

  def test_copy_bufs_pinned_mixed(self):
    bs = [
      [b(0, pin=True), b(1), b(2)],
      [b(1), b(3), b(2)],
      [b(4), b(3)],
      [b(5), b(4), b(0)],
    ]
    check_assign(bs, copies=[(b(1), b(0)), (b(3), b(1))])

  def test_deferred_copy_frees_chain(self):
    bs = []
    copies = []
    for i in range(6):
      copy_buf, compute_buf = b(i * 2 + 1), b(i * 2 + 2)
      bs.append([copy_buf, b(0, pin=True)])
      bs.append([compute_buf, copy_buf])
      copies.append((copy_buf, b(0, pin=True)))
    bs.append([b(100, pin=True)])
    check_assign(bs, copies=copies)

if __name__ == "__main__":
  unittest.main()
