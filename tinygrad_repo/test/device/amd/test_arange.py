import unittest
from tinygrad import Tensor, Device, dtypes, GlobalCounters
from tinygrad.helpers import DEV
from tinygrad.engine.realize import compile_linear, estimate_uop
from test.helpers import KernelCountException

class TestIndexing(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "AMD" or (Device.DEFAULT == "NULL" and DEV.arch.startswith("gfx")), "tests AMD bf16 cast overhead")
  def base_test_llama_8b_rope_backward(self, dtype, ops_scale=1):
    from extra.models.llama import precompute_freqs_cis, apply_rotary_emb
    bs, seqlen, dim, n_heads = 1, 512, 256, 4
    head_dim = dim // n_heads
    x = Tensor.randn(bs, seqlen, dim, dtype=dtype)
    wq = Tensor.randn(dim, dim, dtype=dtype)
    freqs_cis = precompute_freqs_cis(head_dim, seqlen).cast(dtype)
    Tensor.realize(x, wq, freqs_cis)
    xq = (x @ wq.T)
    # main llama does not fuse it
    #xq = xq.contiguous_backward()
    xq = xq.reshape(bs, seqlen, n_heads, head_dim)
    xq_rope, _ = apply_rotary_emb(xq, xq, freqs_cis)
    xq_rope.sum().backward()
    linear = compile_linear(wq.grad.schedule_linear())
    if len(linear.src) != 1: raise KernelCountException(1, len(linear.src))
    bwd_ops = estimate_uop(linear.src[0]).ops
    expected_ops = bs*seqlen*dim*dim*ops_scale
    print(f"rope matmul bwd ({dtype}): {GlobalCounters.kernel_count} kernels, {bwd_ops:,} ops")
    self.assertLess(bwd_ops, expected_ops, f"rope bwd ops {bwd_ops:,} should be < {ops_scale} per (got {bwd_ops/(bs*seqlen*dim*dim):.1f})")

  @unittest.skipIf(DEV.renderer == "LLVM", "TODO: LLVM lowering exceeds the HIP kernel op-count bound")
  def test_llama_8b_rope_backward_f16(self):
    self.base_test_llama_8b_rope_backward(dtypes.float16, ops_scale=2)
  # bfloat16 on non CDNA4 has ~10x ops overhead because of the software emulation
  def test_llama_8b_rope_backward_bf16(self):
    self.base_test_llama_8b_rope_backward(dtypes.bfloat16, ops_scale=2 if Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950") else 25)

if __name__ == "__main__": unittest.main()
