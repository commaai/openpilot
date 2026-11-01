import unittest, functools
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import DEV
from extra.models.llama import precompute_freqs_cis
from extra.thunder.amd.fa import custom_fused_qkv_rope_backward
from test.device.amd.test_asm_gemm import has_hipcc, is_cdna4
from test.runtime.test_llama_kernels import run_swiglu

class TestFusedQKVRoPE(unittest.TestCase):
  SHAPE = (2, 8192, 32, 8, 128)

  def setUp(self):
    if dtypes.bfloat16 not in Device[Device.DEFAULT].renderer.supported_dtypes(): self.skipTest("test uses bf16 inputs")

  def rand_bf16(self, *shape:int) -> Tensor:
    return (Tensor.randn(*shape) * 0.1).cast(dtypes.bfloat16).contiguous().realize()

  @unittest.skipUnless(has_hipcc() and is_cdna4(), "backward kernel requires hipcc to compile")
  @unittest.skipIf(DEV.interface.startswith("MOCK"), "large backward kernel requires real hardware")
  def test_llama31_8b(self):
    Tensor.manual_seed(1)
    B, N, H, H_KV, D = self.SHAPE
    PARTIALS = 2
    GROUP = H // H_KV
    freqs_cis = precompute_freqs_cis(D, N * 2).cast(dtypes.bfloat16).clone().realize()
    dq = self.rand_bf16(B, N, H, D)
    dk_partial = self.rand_bf16(B * PARTIALS, N, H_KV, D)
    dv_partial = self.rand_bf16(B * PARTIALS, N, H_KV, D)

    # Invert Flash Attention's dQ layout transform to reproduce its native buffer.
    dq_native = dq.transpose(1, 2).reshape(B, H, N//16, 4, 4, 4, 2, D//32, 2, 2) \
      .permute(0, 1, 2, 5, 6, 8, 7, 3, 4, 9).reshape(B, H, N, D).contiguous().realize()
    dx = Tensor.empty(B, N, H_KV * (GROUP + 2) * D, dtype=dtypes.bfloat16)
    arch = Device[Device.DEFAULT].renderer.target.arch
    fxn = functools.partial(custom_fused_qkv_rope_backward, device=Device.DEFAULT, arch=arch,
                            B=B, N=N, H=H, H_KV=H_KV, D=D)
    dx = Tensor.custom_kernel(dx, dq_native, dk_partial, dv_partial, freqs_cis, fxn=fxn)[0].realize()

    def inverse_rope(x:Tensor) -> Tensor:
      x = x.reshape(*x.shape[:-1], D//2, 2).float()
      cs = freqs_cis[:, :N].float()
      return Tensor.stack(x[..., 0] * cs[..., 0] + x[..., 1] * cs[..., 1],
                          -x[..., 0] * cs[..., 1] + x[..., 1] * cs[..., 0], dim=-1).flatten(-2).cast(dtypes.bfloat16)

    dq_ref = inverse_rope(dq).reshape(B, N, H_KV, GROUP, D)
    dk_ref = inverse_rope(dk_partial.float().reshape(B, PARTIALS, N, H_KV, D).sum(1).cast(dtypes.bfloat16)).unsqueeze(3)
    dv_ref = dv_partial.float().reshape(B, PARTIALS, N, H_KV, D).sum(1).cast(dtypes.bfloat16).unsqueeze(3)
    ref = Tensor.cat(dq_ref, dk_ref, dv_ref, dim=3).reshape(*dx.shape).realize()
    with Context(DEBUG=0): self.assertTrue(dx.allclose(ref, atol=2e-2, rtol=2e-2).item(), "backward mismatch")

class TestSwiGLU(unittest.TestCase):
  def setUp(self):
    if dtypes.bfloat16 not in Device[Device.DEFAULT].renderer.supported_dtypes(): self.skipTest("need bfloat16")

  def test_llama_shape(self):
    if Device.DEFAULT != "AMD" or DEV.interface.startswith("MOCK") or not Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950"):
      self.skipTest("only run on real machine for speed")
    run_swiglu(self, (2, 8192, 28672))

if __name__ == "__main__": unittest.main()
