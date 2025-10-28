import unittest
from tinygrad import Tensor, dtypes, TinyJit, UOp
from tinygrad.apps.llm import apply_rope

# TODO: test_scheduler, but just in uint
class TestAttention(unittest.TestCase):
  def test_half_qkv_buffers(self):
    BS, seqlen, dim = 10, 4, 100
    q = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    k = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    v = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    attn = q.scaled_dot_product_attention(k, v)
    sched = attn.schedule()
    # attention has 5 kernels now
    self.assertEqual(len(sched), 5)
    softmax_inputs = sched[1:4]
    for si in softmax_inputs:
      assert all(b.dtype == dtypes.half for b in si.bufs), f"non half {si.bufs=}"

  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  def test_apply_rope_jit_prune(self):
    def rope_fn(x_in, pos): return apply_rope(x_in, pos)
    rope_noprune = TinyJit(rope_fn)
    rope_prune = TinyJit(rope_fn, prune=True)

    v_pos = UOp.variable("start_pos", 0, 100)
    for _ in range(3):
      rope_noprune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
      rope_prune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
    noprune_size = len(rope_noprune.captured.jit_cache)
    prune_size = len(rope_prune.captured.jit_cache)

    self.assertGreater(noprune_size, prune_size)
    self.assertGreaterEqual(noprune_size, 3)
    self.assertEqual(prune_size, 1)

if __name__ == '__main__':
  unittest.main()