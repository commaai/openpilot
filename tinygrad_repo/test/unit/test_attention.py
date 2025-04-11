import unittest
from tinygrad import Tensor, dtypes

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

if __name__ == '__main__':
  unittest.main()