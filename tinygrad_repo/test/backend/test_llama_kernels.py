import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8
from extra.llama_kernels.fused_ce import fused_ce_loss
from extra.llama_kernels.quantize_fp8_delayed import quantize_fp8_delayed, quantize_fp8_scalar
from test.helpers import needs_second_gpu

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
    fp8, inv_scale, new_amax, _ = quantize_fp8_delayed(x, amax_state, FP8_DTYPE)
    ref_fp8, ref_inv_scale, ref_new_amax = quantize_fp8(x, amax_state=amax_state)
    Tensor.realize(fp8, inv_scale, new_amax)
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
      assert new_amax.allclose(ref_new_amax, atol=0, rtol=0).item(), \
        f"amax mismatch: got={new_amax.item()} ref={ref_new_amax.item()} diff={abs(new_amax.item()-ref_new_amax.item())}"

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
    fp8, _, new_amax, _ = quantize_fp8_delayed(x, amax_state, FP8_DTYPE)
    Tensor.realize(fp8, new_amax)
    assert fp8.uop.shape == x.uop.shape
    assert new_amax.shape == ()

if __name__ == '__main__':
  unittest.main()
