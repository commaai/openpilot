import unittest
from tinygrad import dtypes, Device
from tinygrad.device import Buffer
from tinygrad.engine.memory import _internal_memory_planner

global_map = {}
def b(i, base=None, offset=0, pin=False, size=16):
  global global_map
  if i in global_map: return global_map[i]
  global_map[i] = Buffer(Device.DEFAULT, size, dtypes.int8, base=global_map[base] if base is not None else None, offset=offset)
  if pin: global_map[i].ref(1)
  return global_map[i]

def check_assign(buffers:list[list[Buffer]|tuple[Buffer, ...]]):
  assigned = _internal_memory_planner(buffers, noopt_buffers=None)

  taken_parts = set()
  first_appearance, last_appearance = {}, {}
  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0: continue
      if buf.base not in first_appearance: first_appearance[buf.base] = i
      last_appearance[buf.base] = i

  for i,u in enumerate(buffers):
    for buf in u:
      if buf.is_allocated() or buf.base.is_allocated() or buf.uop_refcount > 0: continue
      cur, base = assigned.get(buf, buf), assigned.get(buf.base, buf.base)
      if buf._base is not None:
        assert cur.base == base.base and cur.offset == buf.offset + base.offset, f"failed: {buf} {cur} {base} {buf.offset} {base.offset}"
      else:
        for part in taken_parts:
          assert buf.base == part[3] or part[0] != cur.base or part[1] + part[2] <= cur.offset or part[1] >= cur.offset + buf.nbytes
        if first_appearance[buf.base] == i: taken_parts.add((cur.base, cur.offset, buf.nbytes, buf.base))
        if last_appearance[buf.base] == i: taken_parts.remove((cur.base, cur.offset, buf.nbytes, buf.base))

class TestMemoryPlanner(unittest.TestCase):
  def setUp(self):
    global global_map
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

if __name__ == "__main__":
  unittest.main()
