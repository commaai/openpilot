import unittest, functools
from tinygrad import Tensor, Device, dtypes, Context, GlobalCounters
from tinygrad.helpers import getenv
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8
from extra.llama_kernels.fused_ce import fused_ce_loss
from extra.llama_kernels import local_abs_max
from extra.llama_kernels.quantize_fp8_delayed import quantize_fp8_delayed, quantize_fp8_scalar
from extra.models.llama import apply_rotary_emb, precompute_freqs_cis
from extra.thunder.amd.fa import custom_fused_qkv_rope_backward, fused_qkv_rope
from test.helpers import needs_second_gpu
from test.backend.test_asm_gemm import has_hipcc

def run_fused_ce(bs:int, seqlen:int, vocab:int, label_smoothing:float=0.0) -> None:
  Tensor.manual_seed(0)
  logits_rand = Tensor.randn(bs, seqlen, vocab).cast(dtypes.bfloat16)
  targets = Tensor.randint(bs, seqlen, high=vocab, dtype=dtypes.int32)
  logits, logits_ref = logits_rand.clone(), logits_rand.detach().float().contiguous()
  with Context(DEBUG=0):
    Tensor.realize(logits, logits_ref, targets)

  loss = fused_ce_loss(logits, targets, label_smoothing=label_smoothing)
  loss.backward()
  Tensor.realize(loss, logits.grad)

  ref = logits_ref.sparse_categorical_crossentropy(targets, label_smoothing=label_smoothing)
  ref.backward()
  Tensor.realize(ref, logits_ref.grad)

  assert logits.grad.shape == (bs, seqlen, vocab)
  with Context(DEBUG=0):
    assert loss.allclose(ref, atol=2e-3, rtol=2e-3).item(), "forward mismatch"
    assert logits.grad.allclose(logits_ref.grad, atol=2e-3, rtol=2e-3).item(), "grad mismatch"

class TestFusedCE(unittest.TestCase):
  def setUp(self):
    if dtypes.bfloat16 not in Device[Device.DEFAULT].renderer.supported_dtypes(): self.skipTest("need bfloat16")

  def test_fused_ce_1_2_16(self): run_fused_ce(1, 2, 16, label_smoothing=0.2)
  def test_fused_ce_2_16_128(self): run_fused_ce(2, 16, 128)
  def test_fused_ce_4_128_1024(self): run_fused_ce(4, 128, 1024, label_smoothing=0.2)

  # note: this is the shape used in llama 8b
  #def test_fused_ce_smoothing_16_1024_128256(self): run_fused_ce(16, 1024, 128256, label_smoothing=0.2)

def run_quantize_fp8(shape:tuple[int, ...], delayed:bool=True) -> None:
  Tensor.manual_seed(0)
  x = Tensor.randn(*shape).cast(dtypes.bfloat16).contiguous()
  amax_state = Tensor.full((), 2.0, dtype=dtypes.float32).contiguous()
  with Context(DEBUG=0): Tensor.realize(x, amax_state)

  if delayed:
    amax_out = Tensor.zeros((), dtype=dtypes.float32, device=x.device).realize()
    fp8, inv_scale = quantize_fp8_delayed(x, amax_state, amax_out, FP8_DTYPE)
    ref_fp8, ref_inv_scale, ref_new_amax = quantize_fp8(x, amax_state=amax_state)
    Tensor.realize(fp8, inv_scale)
    Tensor.realize(ref_fp8, ref_inv_scale, ref_new_amax)
  else:
    fp8 = quantize_fp8_scalar(x, amax_state, FP8_DTYPE)
    ref_fp8, _, _ = quantize_fp8(x, amax_state=amax_state)
    Tensor.realize(fp8)
    Tensor.realize(ref_fp8)

  with Context(DEBUG=0):
    assert fp8.cast(dtypes.float).allclose(ref_fp8.cast(dtypes.float), atol=0, rtol=0).item(), "fp8 mismatch"
    if delayed:
      assert inv_scale.allclose(ref_inv_scale, atol=0, rtol=0).item(), "inv_scale mismatch"
      assert amax_out.allclose(ref_new_amax, atol=0, rtol=0).item(), \
        f"amax mismatch: got={amax_out.item()} ref={ref_new_amax.item()} diff={abs(amax_out.item()-ref_new_amax.item())}"

@unittest.skipUnless(Device.DEFAULT == "AMD", "requires atomic max")
class TestQuantizeFP8(unittest.TestCase):
  def setUp(self):
    ren = Device[Device.DEFAULT].renderer
    if dtypes.bfloat16 not in ren.supported_dtypes(): self.skipTest("need bfloat16")
    if not ren.has_local or not ren.has_shared: self.skipTest("need local/shared")

  def test_scalar(self): run_quantize_fp8((getenv("N", 1024), 32), delayed=False)
  def test_delayed(self): run_quantize_fp8((getenv("N", 2048), 1024))

  @needs_second_gpu
  def test_multi(self):
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(8))
    x = Tensor.empty(2048*8, 1024, dtype=dtypes.bfloat16, device=devs).uop.multi(0)
    x = Tensor(x, device=devs)
    amax_state = Tensor.full((), 2.0, dtype=dtypes.float32, device=devs).contiguous()
    amax_out = Tensor.zeros((), dtype=dtypes.float32, device=devs).realize()
    fp8, _ = quantize_fp8_delayed(x, amax_state, amax_out, FP8_DTYPE)
    Tensor.realize(fp8)
    assert fp8.uop.shape == x.uop.shape
    assert amax_out.shape == ()

class TestLocalAmax(unittest.TestCase):
  def test_multi_tensor_local_shard_amax(self):
    devices = ("CPU:0", "CPU:1")
    x = Tensor.arange(16).reshape(4, 4).cast(dtypes.float).clone(devices[0]).realize().shard(devices, axis=0).realize()
    GlobalCounters.reset()
    out = (x * local_abs_max(x)).clone().realize()
    self.assertEqual(GlobalCounters.kernel_count, 2)
    self.assertEqual(out.tolist(), [[0., 7., 14., 21.], [28., 35., 42., 49.], [120., 135., 150., 165.], [180., 195., 210., 225.]])

@unittest.skipUnless(has_hipcc() and Device.DEFAULT == "AMD", "requires hipcc to compile and amd device to run")
class TestFusedQKVRoPE(unittest.TestCase):
  SHAPE = (2, 8192, 32, 8, 128)

  def rand_bf16(self, *shape:int) -> Tensor:
    return (Tensor.randn(*shape) * 0.1).cast(dtypes.bfloat16).contiguous().realize()

  def freqs_cis(self) -> Tensor:
    _, N, _, _, D = self.SHAPE
    return precompute_freqs_cis(D, N * 2).cast(dtypes.bfloat16).clone().realize()

  def test_llama31_8b_forward(self):
    Tensor.manual_seed(0)
    B, N, H, H_KV, D = self.SHAPE
    GROUP = H // H_KV
    freqs_cis = self.freqs_cis()

    x = self.rand_bf16(B, N, H_KV * (GROUP + 2) * D)
    q, k, v = fused_qkv_rope(x, freqs_cis, H, H_KV, D)
    Tensor.realize(q, k, v)
    packed_ref = x.reshape(B, N, H_KV, GROUP + 2, D)
    q_ref = packed_ref[:, :, :, :GROUP].reshape(B, N, H, D)
    k_ref, v_ref = packed_ref[:, :, :, GROUP], packed_ref[:, :, :, GROUP+1]
    q_ref, k_ref = apply_rotary_emb(q_ref, k_ref, freqs_cis[:, :N])
    q_ref, k_ref, v_ref = q_ref.cast(dtypes.bfloat16), k_ref.cast(dtypes.bfloat16), v_ref.cast(dtypes.bfloat16)
    Tensor.realize(q_ref, k_ref, v_ref)

    with Context(DEBUG=0):
      self.assertTrue(q.allclose(q_ref, atol=2e-2, rtol=0).item(), "Q forward mismatch")
      self.assertTrue(k.allclose(k_ref, atol=2e-2, rtol=0).item(), "K forward mismatch")
      self.assertTrue(v.allclose(v_ref, atol=0, rtol=0).item(), "V forward mismatch")

  def test_llama31_8b_backward(self):
    Tensor.manual_seed(1)
    B, N, H, H_KV, D = self.SHAPE
    PARTIALS = 2
    GROUP = H // H_KV
    freqs_cis = self.freqs_cis()
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

if __name__ == '__main__':
  unittest.main()
