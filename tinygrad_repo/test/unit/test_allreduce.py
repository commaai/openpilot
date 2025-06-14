import unittest
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops

class TestRingAllReduce(unittest.TestCase):
  def test_schedule_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      schedules = t.sum(0).schedule_with_vars()[0]
      copies = [si for si in schedules if si.ast.op is Ops.COPY]
      pairs = [(c.bufs[0].device, c.bufs[1].device) for c in copies]
      # N*(N-1) scatter reduce, and N*(N-1) allgather
      self.assertEqual(len(pairs), N*(N-1)*2)
      # copy topology forms a ring
      self.assertEqual(len(set(pairs)), N)

  def test_correct_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.ones(N, N*100).contiguous().shard(ds, axis=0).realize()
      out = t.sum(0)
      self.assertListEqual(out.tolist(), [4]*N*100)

if __name__ == '__main__':
  unittest.main()
