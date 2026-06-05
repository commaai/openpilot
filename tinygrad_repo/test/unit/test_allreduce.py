import unittest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops

class TestRingAllReduce(unittest.TestCase):
  def test_schedule_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      linear = t.sum(0).linear_with_vars()[0]
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]
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

class TestAllreduceCast(unittest.TestCase):
  def _get_copy_dtypes(self, dtype, allreduce_cast):
    ds = tuple(f"CPU:{i}" for i in range(2))
    with Context(ALLREDUCE_CAST=allreduce_cast, RING=0, SCACHE=0):
      t = Tensor.empty(4, 4, dtype=dtype).shard(ds, axis=0)
      linear = t.sum(0).linear_with_vars()[0]
      return {si.src[1].buffer.dtype.scalar() for si in linear.src if si.src[0].op is Ops.COPY}

  def test_allreduce_cast_bf16(self):
    # with ALLREDUCE_CAST, allreduce copies stay in bfloat16 instead of promoting to float32
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=0))

  def test_allreduce_cast_half(self):
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=0))

  def test_allreduce_cast_float32_noop(self):
    # float32 should not be affected by ALLREDUCE_CAST (no promotion happens)
    dtypes_on = self._get_copy_dtypes(dtypes.float, allreduce_cast=1)
    dtypes_off = self._get_copy_dtypes(dtypes.float, allreduce_cast=0)
    self.assertEqual(dtypes_on, dtypes_off)

if __name__ == '__main__':
  unittest.main()
