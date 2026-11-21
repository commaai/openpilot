#!/usr/bin/env python
import unittest
from tinygrad import dtypes, Tensor, GlobalCounters
from test.helpers import assert_kernel_count
N = 200  # has to be bigger than the cache to fail

class TestAssign(unittest.TestCase):
  def test_assign_contiguous(self):
    b = Tensor.arange(16).reshape(4,4).clone().realize()
    a = (Tensor.arange(16).reshape(4,4).clone().realize() + 1)
    GlobalCounters.reset()
    b.assign(a.contiguous()).realize()
    assert_kernel_count(2)

  def test_assign_contiguous_permute(self):
    b = Tensor.arange(16).reshape(4,4).clone().realize()
    a = (Tensor.arange(16).reshape(4,4).clone().realize() + 1).permute((1,0))
    GlobalCounters.reset()
    b.assign(a.contiguous()).realize()
    assert_kernel_count(2)

  # IEEE 754: 1.0f = 0x3f800000, 2.0f = 0x40000000, 3.0f = 0x40400000, 4.0f = 0x40800000
  REVERSED = [0x40800000, 0x40400000, 0x40000000, 0x3f800000]

  def test_assign_dtype_mismatch(self):
    # assign should not implicitly cast dtypes - this can lose precision
    a = Tensor.zeros(4, dtype=dtypes.float32).contiguous().realize()
    b = Tensor([1, 2, 3, 4], dtype=dtypes.int32)
    with self.assertRaisesRegex(RuntimeError, "assign dtype mismatch"):
      a.assign(b)

  def test_chained_assign_kernel_count(self):
    """Chained pending assigns must not produce excessive kernels (tests recursive transitive processing)."""
    D, N = 4, 5
    caches = [Tensor.zeros(8, D).contiguous().realize() for _ in range(N)]
    caches[0][0:1].assign(Tensor.ones(1, D, buffer=False) * 10)
    x = caches[0][:1].sum(0, keepdim=True)
    for i in range(1, N):
      caches[i][0:1].assign(x)
      x = caches[i][:1].sum(0, keepdim=True)
    GlobalCounters.reset()
    x.realize()
    # N assigns (1 kernel each) producing N kernels total
    assert_kernel_count(N)

  def test_shared_computation_assign_kernel_count(self):
    """When a .contiguous() is shared between an assign value and the next layer's input (like QKV projection in LLM),
    substitute optimization replaces already-realized sub-graphs in remaining pending assigns, preventing kernel escalation.
    Without substitute, pending assign graphs grow linearly and produce 153 kernels instead of 48."""
    D, N = 16, 16
    caches = [Tensor.zeros(4, D).contiguous().realize() for _ in range(N)]
    W = [Tensor.full((D, D*2), 0.01).contiguous().realize() for _ in range(N)]
    x = Tensor.ones(1, D).contiguous().realize()
    for i in range(N):
      shared = (x @ W[i]).contiguous()  # .contiguous() UOp is shared between assign (k) and next layer (q)
      k, q = shared[:, :D], shared[:, D:]
      caches[i][0:1].assign(k)          # assign references the CONTIGUOUS
      x = q + caches[i][:1]             # next layer also references the same CONTIGUOUS through q
    GlobalCounters.reset()
    caches[-1][:1].contiguous().realize()
    # N matmuls + N assigns + 1 final read = 2*N+1 (AFTER embedding allows full graph scheduling with shared contiguous reuse)
    assert_kernel_count(2*N+1)

if __name__ == '__main__':
  unittest.main()
