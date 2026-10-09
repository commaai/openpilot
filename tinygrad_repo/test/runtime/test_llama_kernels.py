import unittest
from tinygrad import Tensor, Device, dtypes, Context, GlobalCounters
from extra.llama_kernels.fused_ce import fused_ce_loss
from extra.llama_kernels import local_abs_max
from extra.llama_kernels.swiglu import swiglu
from extra.models.llama import apply_rotary_emb
from extra.thunder.amd.fa import fused_qkv_rope
from test.helpers import assert_kernel_count

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

class TestLocalAmax(unittest.TestCase):
  def test_multi_tensor_local_shard_amax(self):
    devices = ("CPU:0", "CPU:1")
    x = Tensor.arange(16).reshape(4, 4).cast(dtypes.float).clone(devices[0]).realize().shard(devices, axis=0).realize()
    GlobalCounters.reset()
    out = (x * local_abs_max(x)).clone().realize()
    assert_kernel_count(2)
    self.assertEqual(out.tolist(), [[0., 7., 14., 21.], [28., 35., 42., 49.], [120., 135., 150., 165.], [180., 195., 210., 225.]])

class TestFusedQKVRoPE(unittest.TestCase):
  def setUp(self):
    if dtypes.bfloat16 not in Device[Device.DEFAULT].renderer.supported_dtypes(): self.skipTest("test uses bf16 inputs")

  def rand_bf16(self, *shape:int) -> Tensor:
    return (Tensor.randn(*shape) * 0.1).cast(dtypes.bfloat16).contiguous().realize()

  def test_forward(self):
    Tensor.manual_seed(0)
    B, N, H, H_KV, D = 1, 32, 8, 2, 16
    GROUP = H // H_KV
    freqs_cis = (Tensor.randn(1, N * 2, 1, D // 2, 2) * 0.1).cast(dtypes.bfloat16).contiguous().realize()

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

def run_swiglu(test:unittest.TestCase, shape:tuple[int, ...]) -> None:
  Tensor.manual_seed(0)
  x = (Tensor.randn(*shape) * 2).cast(dtypes.bfloat16).realize()
  hidden = x.shape[-1] // 2
  out, ref = swiglu(x), x[..., :hidden].silu() * x[..., hidden:]
  Tensor.realize(out, ref)
  with Context(DEBUG=0): test.assertTrue(out.allclose(ref, atol=2.5e-1, rtol=3e-2).item(), "SwiGLU forward mismatch")

  grad = (Tensor.randn(*out.shape) * 2).cast(dtypes.bfloat16).realize()
  grad_x, grad_ref = out.gradient(x, gradient=grad)[0], ref.gradient(x, gradient=grad)[0]
  Tensor.realize(grad_x, grad_ref)
  test.assertEqual(grad_x.shape, shape)
  test.assertEqual(grad_x.dtype, dtypes.bfloat16)
  with Context(DEBUG=0): test.assertTrue(grad_x.allclose(grad_ref, atol=2.5e-1, rtol=3e-2).item(), "SwiGLU backward mismatch")

class TestSwiGLU(unittest.TestCase):
  def setUp(self):
    if dtypes.bfloat16 not in Device[Device.DEFAULT].renderer.supported_dtypes(): self.skipTest("need bfloat16")

  def test_simple(self): run_swiglu(self, (2, 32, 64))

if __name__ == '__main__':
  unittest.main()
