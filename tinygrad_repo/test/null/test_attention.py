import unittest
from tinygrad import Tensor, dtypes, TinyJit, UOp
from tinygrad.apps.llm import apply_rope as apply_rope_new, precompute_freqs_cis

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestAttention(unittest.TestCase):
  def test_half_qkv_buffers(self):
    BS, seqlen, dim = 10, 4, 100
    q = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    k = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    v = Tensor.ones(BS, seqlen, dim, dtype=dtypes.half).contiguous().realize()
    attn = q.scaled_dot_product_attention(k, v)
    sched = attn.schedule()
    # attention has 4 kernels now
    self.assertEqual(len(sched), 4)

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
    self.assertGreaterEqual(noprune_size, 2)
    self.assertEqual(prune_size, 1)

if __name__ == '__main__':
  unittest.main()
