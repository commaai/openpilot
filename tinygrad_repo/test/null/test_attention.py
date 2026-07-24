import unittest
from tinygrad import Tensor, dtypes, TinyJit, UOp
from tinygrad.llm.model import apply_rope as apply_rope_new, precompute_freqs_cis
from test.helpers import assert_jit_cache_len

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
    sched = attn.schedule_linear()
    # attention has 4 kernels now
    self.assertEqual(len(sched.src), 4)

  def test_apply_rope_jit_prune(self):
    def rope_fn(x_in, pos): return apply_rope(x_in, pos)
    rope_noprune = TinyJit(rope_fn)
    rope_prune = TinyJit(rope_fn, prune=True)

    v_pos = UOp.variable("start_pos", 0, 100)
    for _ in range(3):
      rope_noprune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
      rope_prune(Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32), v_pos.bind(1))
    assert_jit_cache_len(rope_prune, 1)
    assert_jit_cache_len(rope_noprune, 3)

if __name__ == '__main__':
  unittest.main()
