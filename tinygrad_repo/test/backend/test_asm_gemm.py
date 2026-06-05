import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv, system, DEV
from extra.gemm.cdna_asm_gemm import asm_gemm
from test.helpers import needs_second_gpu
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8, FP8_MAX

# On non CDNA4 it will only validate the Tensor.custom_kernel integration
# Use DEV=NULL:HIP:gfx950 to also test the assembly
def is_cdna4(): return Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950")

def run_asm_gemm(a_shape, b_shape, dtype=dtypes.float16, a_shard=None, b_shard=None, gpus:int=1) -> None:
  Tensor.manual_seed(0)
  input_dtype = dtypes.bfloat16 if dtype == FP8_DTYPE else dtype
  a_rand = Tensor.randn(a_shape, dtype=dtypes.float).sub(0.5).cast(input_dtype)
  b_rand = Tensor.randn(b_shape, dtype=dtypes.float).sub(0.5).cast(input_dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  if dtype == FP8_DTYPE:
    a_rand, x_scale, _ = quantize_fp8(a_rand)
    b_rand, w_scale, _ = quantize_fp8(b_rand)
    grad_amax_state = Tensor.full((), FP8_MAX, dtype=dtypes.float32, device=devs).contiguous()
    with Context(DEBUG=0):
      Tensor.realize(a_rand, x_scale, b_rand, w_scale, grad_amax_state)

  # clone all inputs before any backward: a clone copies the source's current .grad
  a, b = a_rand.clone(), b_rand.clone()
  if dtype == FP8_DTYPE:
    a_ref, b_ref = a_rand.detach().cast(dtypes.bfloat16), b_rand.detach().cast(dtypes.bfloat16)
  else:
    a_ref, b_ref = a_rand.clone(), b_rand.clone()
  if multi: a, b = a.shard(devs, axis=a_shard), b.shard(devs, axis=b_shard)
  if dtype == FP8_DTYPE:
    tst = asm_gemm(a, b, x_scale=x_scale, w_scale=w_scale, grad_amax_state=grad_amax_state)
  else:
    tst = asm_gemm(a, b)
  tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  if multi: a_ref, b_ref = a_ref.shard(devs, axis=a_shard), b_ref.shard(devs, axis=b_shard)
  if dtype == FP8_DTYPE:
    ref = ((a_ref @ b_ref) * x_scale * w_scale).cast(dtypes.bfloat16)
  else:
    ref = a_ref @ b_ref
  ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  # no validation on the NULL device
  if a_rand.device.startswith("NULL"): return None
  atol, rtol = (2e-1, 1e-2) if dtype == dtypes.bfloat16 else (256, 1e-2) if dtype == FP8_DTYPE else (1e-2, 1e-3)
  # allow more rtol for multi because of ALLREDUCE_CAST
  grad_atol, grad_rtol = (16895, 0.125) if dtype == FP8_DTYPE else (atol, 2e-2 if multi else rtol)
  with Context(DEBUG=0):
    # enable for debugging, slow for larger gemms
    if getenv("USE_NPY"):
      import numpy as np
      np.testing.assert_allclose(tst.numpy(), ref.numpy(), atol=atol, rtol=rtol)
      np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), atol=grad_atol, rtol=grad_rtol)
      np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), atol=grad_atol, rtol=grad_rtol)
    assert tst.allclose(ref, atol=atol, rtol=rtol).item(), "forward mismatch"
    assert a.grad.allclose(a_ref.grad, atol=grad_atol, rtol=grad_rtol).item(), "grad_a mismatch"
    assert b.grad.allclose(b_ref.grad, atol=grad_atol, rtol=grad_rtol).item(), "grad_b mismatch"

def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=1) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_k_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=8) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus)

def verify_asm_gemm_n_sharded(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_m_sharded(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_n_sharded_2d(M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_k_sharded_3d(batch:int, M:int, N:int, K:int, dtype=dtypes.float16, gpus:int=2) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=2, b_shard=0, gpus=gpus)

# 128x smaller than usual
# uses the UOp GEMM, runs on non CDNA4 and CI
@unittest.skipUnless(dtypes.half in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
class TestGemm(unittest.TestCase):
  def setUp(self):
    if is_cdna4(): self.skipTest("shapes are too small for the assembly GEMM")
  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 32), N, N, dtype=dtypes.half)
  def test_gemm(self): verify_asm_gemm(1, 64, 32, 112)
  def test_gemm_batched(self): verify_asm_gemm(2, 64, 32, 32)
  @needs_second_gpu
  def test_gemm_multi(self): verify_asm_gemm(2, 64, 32, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_k_sharded(self): verify_asm_gemm_k_sharded(64, 64, 2*64, gpus=2)
  @needs_second_gpu
  def test_gemm_m_sharded(self): verify_asm_gemm_m_sharded(2*64, 64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_n_sharded(self): verify_asm_gemm_n_sharded(1, 64, 64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_n_sharded_2d(self): verify_asm_gemm_n_sharded_2d(64, 2*64, 32, gpus=2)
  @needs_second_gpu
  def test_gemm_k_sharded_3d(self): verify_asm_gemm_k_sharded_3d(1, 64, 32, 2*64, gpus=2)

# uses the smallest size for the cdna assembly gemm
class TestAsmGEMM(unittest.TestCase):
  def setUp(self):
    if not is_cdna4():
      self.skipTest("assembly gemm is only for cdna4")

  def test_tiny(self): verify_asm_gemm(1, 256, 256, 64)

  def test_verify_with_numpy(self):
    import numpy as np
    M, N, K = 256, 256, 64
    rng = np.random.default_rng(0)
    a_np = (rng.random((M, K), dtype=np.float32) - 0.5).astype(np.half)
    b_np = (rng.random((K, N), dtype=np.float32) - 0.5).astype(np.half)
    c_np = a_np @ b_np
    a, b = Tensor(a_np), Tensor(b_np)
    c = asm_gemm(a, b)
    c.realize()
    # no validation on the NULL device
    if a.device.startswith("NULL"): return None
    np.testing.assert_allclose(c.numpy(), c_np, atol=2e-3, rtol=5e-2)

  def test_unsupported_batch(self):
    with self.assertRaisesRegex(AssertionError, "batch size"):
      verify_asm_gemm(3, 256, 256, 256)

  def test_unsupported_k(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1024, 1024, 100)
  def test_unsupported_m(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 1000, 256, 256)
  def test_unsupported_n(self):
    with self.assertRaisesRegex(AssertionError, "not a multiple"):
      verify_asm_gemm(1, 256, 1000, 256)

# test the Asm GEMM with Llama shapes, only run on the real machine for speed
class TestGemmLlama(unittest.TestCase):
  dtype = dtypes.bfloat16

  def setUp(self):
    if not is_cdna4() or DEV.interface.startswith("MOCK"):
      self.skipTest("very slow on non mi350x")

  def test_empty(self): asm_gemm(Tensor.empty(N:=getenv("N", 4096), N, dtype=self.dtype), Tensor.empty(N, N, dtype=self.dtype)).realize()

  def test_empty_bw(self):
    x = Tensor.empty(1, N:=getenv("N", 4096), N, dtype=self.dtype)
    y = Tensor.empty((N, N), dtype=self.dtype)
    if self.dtype == FP8_DTYPE:
      x_scale = Tensor.empty((), dtype=dtypes.float32)
      w_scale = Tensor.empty((), dtype=dtypes.float32)
      grad_amax_state = Tensor.empty((), dtype=dtypes.float32).contiguous()
      z = asm_gemm(x, y, x_scale=x_scale, w_scale=w_scale, grad_amax_state=grad_amax_state)
    else:
      z = asm_gemm(x, y)
    z.sum().backward()
    Tensor.realize(z, x.grad, y.grad)
    # FP8 GEMM stores bf16 output and its backward produces bf16 gradients.
    grad_dtype = dtypes.bfloat16 if self.dtype == FP8_DTYPE else self.dtype
    assert z.dtype == dtypes.bfloat16
    assert x.grad.dtype == y.grad.dtype == grad_dtype

  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 4096), N, N, dtype=self.dtype)
  def test_gemm(self): verify_asm_gemm(1, 8192, 4096, 14336, dtype=self.dtype)
  def test_gemm_batched(self): verify_asm_gemm(2, 8192, 4096, 4096, dtype=self.dtype)

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 14336, dtype=self.dtype, gpus=8)
  @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm2(self): verify_asm_gemm(8, 8192, 128256, 4096, dtype=self.dtype, gpus=8)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_gemm4(self): verify_asm_gemm(8, 4096, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_gemm5(self): verify_asm_gemm(8, 4096, 4096, 14336, dtype=self.dtype, gpus=8)
  def test_gemm6(self): verify_asm_gemm(16, 4096, 4096, 14336, dtype=self.dtype, gpus=8)
  @unittest.skip("disabled, asm in this shape is slower than tinygrad")
  def test_gemm7(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype)
  def test_gemm8(self): verify_asm_gemm(1, 4096, 14336, 8192, dtype=self.dtype)
  def test_gemm9(self): verify_asm_gemm(8, 4096, 14336, 8192, dtype=self.dtype, gpus=8)
  def test_gemm10(self): verify_asm_gemm(1, 4096, 8192, 4096, dtype=self.dtype)
  def test_gemm_previously_unsupported(self): verify_asm_gemm(8, 1024, 1024, 4096, gpus=8)
  def test_k_sharded_1(self): verify_asm_gemm_k_sharded(14336, 4096, 8*8192, dtype=self.dtype, gpus=8)
  def test_k_sharded_2(self): verify_asm_gemm_k_sharded(4096, 14336, 8*8192, dtype=self.dtype, gpus=8)
  def test_k_sharded_3(self): verify_asm_gemm_k_sharded(4096, 4096, 8*8192, dtype=self.dtype, gpus=8)

  # M-sharded 2D
  def test_m_sharded_1(self): verify_asm_gemm_m_sharded(8*8192, 4096, 4096, dtype=self.dtype, gpus=8)
  def test_m_sharded_2(self): verify_asm_gemm_m_sharded(8*4096, 14336, 4096, dtype=self.dtype, gpus=8)

  # N-sharded 2D
  def test_n_sharded_2d_1(self): verify_asm_gemm_n_sharded_2d(8192, 8*4096, 4096, dtype=self.dtype, gpus=8)
  def test_n_sharded_2d_2(self): verify_asm_gemm_n_sharded_2d(4096, 8*14336, 4096, dtype=self.dtype, gpus=8)

  # tensor parallel shapes (Llama 8B, MP=8)
  def test_tp_n_sharded_wq(self): verify_asm_gemm_n_sharded(1, 8192, 4096, 4096, dtype=self.dtype, gpus=8)
  def test_tp_n_sharded_w1(self): verify_asm_gemm_n_sharded(1, 8192, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_tp_k_sharded_wo(self): verify_asm_gemm_k_sharded_3d(1, 8192, 4096, 4096, dtype=self.dtype, gpus=8)
  def test_tp_k_sharded_w2(self): verify_asm_gemm_k_sharded_3d(1, 8192, 4096, 14336, dtype=self.dtype, gpus=8)

  # more shapes: vary M, N, K independently
  def test_shape_small_square(self): verify_asm_gemm(1, 256, 256, 256)
  def test_shape_small_rect_m(self): verify_asm_gemm(1, 512, 256, 256)
  def test_shape_small_rect_n(self): verify_asm_gemm(1, 256, 512, 256)
  def test_shape_small_rect_k(self): verify_asm_gemm(1, 256, 256, 512)
  def test_shape_tall(self): verify_asm_gemm(1, 2048, 256, 256)
  def test_shape_wide(self): verify_asm_gemm(1, 256, 2048, 256)
  def test_shape_deep(self): verify_asm_gemm(1, 256, 256, 4096)
  def test_shape_non_square(self): verify_asm_gemm(1, 1024, 2048, 512)
  def test_shape_batched_small(self): verify_asm_gemm(2, 256, 256, 256)
  def test_shape_batched_rect(self): verify_asm_gemm(2, 512, 1024, 256)
  # K edge cases: iters=1,2,3 exercise different loop paths
  def test_shape_k64(self): verify_asm_gemm(1, 256, 256, 64)
  def test_shape_k128(self): verify_asm_gemm(1, 256, 256, 128)
  def test_shape_k192(self): verify_asm_gemm(1, 256, 256, 192)

  def test_llama3_out1(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype)
  def test_llama3_out2(self): verify_asm_gemm(1, 8192, 4096, 128256, dtype=self.dtype)
  def test_llama3_out3(self): verify_asm_gemm(1, 4096, 128256, 8192, dtype=self.dtype)

def has_hipcc():
  try: system("hipcc --version")
  except Exception: return False
  return True

@unittest.skipUnless(has_hipcc(), "FP8 gemm requires hipcc to compile")
class TestGemmLlamaFP8(TestGemmLlama): dtype = FP8_DTYPE

class TestMagicGu(unittest.TestCase):
  def test_magicgu_matches_old(self):
    from extra.gemm.cdna_asm_gemm import _magicgu_mulhi, TILE_M, TILE_N, TILE_K
    old_iters_args = {64: (67108864, 0), 128: (33554432, 0), 224: (613566757, 2147483656)}
    old_gemm_shapes = [
      (8192, 4096, 4096), (8192, 14336, 4096), (8192, 4096, 14336),
      (8192, 8192, 8192), (4096, 4096, 4096), (4096, 14336, 4096),
      (4096, 14336, 8192), (4096, 4096, 14336), (14336, 4096, 8192),
      (4096, 8192, 14336), (4096, 4096, 8192), (4096, 8192, 4096),
    ]
    for M, N, K in old_gemm_shapes:
      iters = K // TILE_K
      total = (M // TILE_M) * (N // TILE_N) * iters
      for batch in [1, 2]:
        magic, shift = _magicgu_mulhi(iters, total * batch)
        old_magic, old_shift = old_iters_args[iters]
        self.assertEqual((magic, shift), (old_magic, old_shift), f"mismatch for ({M},{N},{K}) batch={batch} iters={iters}")

if __name__ == "__main__":
  unittest.main()
