import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv, system, DEV
from extra.gemm.cdna_asm_gemm import asm_gemm, hk_bf16_atb_gemm
from test.helpers import needs_second_gpu
from examples.mlperf.models.flat_llama import FP8_DTYPE, quantize_fp8, FP8_MAX

# On non CDNA4 it will only validate the Tensor.custom_kernel integration
# Use DEV=NULL:HIP:gfx950 to also test the assembly
def is_cdna4(): return Device[Device.DEFAULT].renderer.target.arch.startswith("gfx950")

def has_hipcc():
  try: system("hipcc --version")
  except Exception: return False
  return True

def run_asm_gemm(a_shape, b_shape, dtype=dtypes.bfloat16, a_shard=None, b_shard=None, gpus:int=1) -> None:
  Tensor.manual_seed(0)
  input_dtype = dtypes.bfloat16 if dtype == FP8_DTYPE else dtype
  a_rand = Tensor.randn(a_shape, dtype=dtypes.float).sub(0.5).cast(input_dtype)
  b_rand = Tensor.randn(b_shape, dtype=dtypes.float).sub(0.5).cast(input_dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  if dtype == FP8_DTYPE:
    x_scale = Tensor.full((), FP8_MAX, dtype=dtypes.float32, device=devs).contiguous()
    a_rand, _, _ = quantize_fp8(a_rand.shard(devs, axis=a_shard) if multi else a_rand, amax_state=x_scale)
    b_rand, w_scale, _ = quantize_fp8(b_rand.T.contiguous())
    if multi: b_rand, w_scale = b_rand.shard(devs, axis=None if b_shard is None else 1-b_shard), w_scale.to(devs).contiguous()
    grad_amax_state = Tensor.full((), FP8_MAX, dtype=dtypes.float32, device=devs).contiguous()
    next_grad_amax_state = Tensor.empty((), dtype=dtypes.float32, device=devs)
    with Context(DEBUG=0):
      Tensor.realize(a_rand, x_scale, b_rand, w_scale, grad_amax_state, next_grad_amax_state)

  # clone all inputs before any backward: a clone copies the source's current .grad
  a, b = a_rand.clone(), b_rand.clone()
  if dtype == FP8_DTYPE:
    a_ref, b_ref = a_rand.detach().cast(dtypes.bfloat16), b_rand.detach().cast(dtypes.bfloat16)
  else:
    a_ref, b_ref = a_rand.clone(), b_rand.clone()
  if multi and isinstance(a.device, str): a, b = a.shard(devs, axis=a_shard), b.shard(devs, axis=b_shard)
  if dtype == FP8_DTYPE:
    tst = asm_gemm(a, b.T, x_scale=x_scale, w_scale=w_scale, grad_amax_state=grad_amax_state,
                   next_grad_amax_state=next_grad_amax_state)
  else:
    tst = asm_gemm(a, b)
  tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  if multi and isinstance(a_ref.device, str): a_ref, b_ref = a_ref.shard(devs, axis=a_shard), b_ref.shard(devs, axis=b_shard)
  if dtype == FP8_DTYPE:
    ref = ((a_ref @ b_ref.T) * ((x_scale.float() + 1e-8) / FP8_MAX) * w_scale).cast(dtypes.bfloat16)
  else:
    ref = a_ref @ b_ref
  ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  # no validation on the NULL device
  if Device.DEFAULT.startswith("NULL"): return None
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

def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=1) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_k_sharded(M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=8) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=1, b_shard=0, gpus=gpus)

def verify_asm_gemm_n_sharded(batch:int, M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=2) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_m_sharded(M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=2) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=0, b_shard=None, gpus=gpus)

def verify_asm_gemm_n_sharded_2d(M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=2) -> None:
  run_asm_gemm((M, K), (K, N), dtype=dtype, a_shard=None, b_shard=1, gpus=gpus)

def verify_asm_gemm_k_sharded_3d(batch:int, M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=2) -> None:
  run_asm_gemm((batch, M, K), (K, N), dtype=dtype, a_shard=2, b_shard=0, gpus=gpus)

# 128x smaller than usual
# uses the UOp GEMM, runs on non CDNA4 and CI
@unittest.skipUnless(dtypes.bfloat16 in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
class TestGemm(unittest.TestCase):
  def setUp(self):
    if is_cdna4(): self.skipTest("shapes are too small for the assembly GEMM")
  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 32), N, N, dtype=dtypes.bfloat16)
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
    if not is_cdna4() or not has_hipcc():
      self.skipTest("assembly gemm is only for cdna4")

  def test_tiny(self): verify_asm_gemm(1, 256, 256, 256)

  def test_verify_with_numpy(self):
    import numpy as np
    M, N, K = 256, 256, 256
    rng = np.random.default_rng(0)
    a_np = (rng.random((M, K), dtype=np.float32) - 0.5).astype(np.float32)
    b_np = (rng.random((K, N), dtype=np.float32) - 0.5).astype(np.float32)
    c_np = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float32)
    Tensor.manual_seed(0)
    a, b = Tensor(a_np).cast(dtypes.bfloat16), Tensor(b_np).cast(dtypes.bfloat16)
    c = asm_gemm(a, b)
    c.realize()
    # no validation on the NULL device
    if a.device.startswith("NULL"): return None
    np.testing.assert_allclose(c.numpy(), c_np, atol=2e-1, rtol=1e-2)

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

@unittest.skipUnless(has_hipcc(), "requires hipcc to compile")
class TestGemmLlama(unittest.TestCase):
  dtype = FP8_DTYPE

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
      next_grad_amax_state = Tensor.empty((), dtype=dtypes.float32)
      z = asm_gemm(x, y, x_scale=x_scale, w_scale=w_scale, grad_amax_state=grad_amax_state,
                   next_grad_amax_state=next_grad_amax_state)
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
  def test_gemm2(self): verify_asm_gemm(8, 8192, 128256, 4096, dtype=self.dtype, gpus=8)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_gemm4(self): verify_asm_gemm(8, 4096, 14336, 4096, dtype=self.dtype, gpus=8)
  def test_gemm5(self): verify_asm_gemm(8, 4096, 4096, 14336, dtype=self.dtype, gpus=8)
  def test_gemm6(self): verify_asm_gemm(16, 4096, 4096, 14336, dtype=self.dtype, gpus=8)
  def test_gemm7(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype)
  def test_gemm8(self): verify_asm_gemm(1, 4096, 14336, 8192, dtype=self.dtype)
  def test_gemm9(self): verify_asm_gemm(8, 4096, 14336, 8192, dtype=self.dtype, gpus=8)
  def test_gemm10(self): verify_asm_gemm(1, 4096, 8192, 4096, dtype=self.dtype)
  def test_gemm11(self): verify_asm_gemm(8, 1024, 1024, 4096, dtype=self.dtype, gpus=8)
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
  def test_shape_small_square(self): verify_asm_gemm(1, 256, 256, 256, dtype=self.dtype)
  def test_shape_small_rect_m(self): verify_asm_gemm(1, 512, 256, 256, dtype=self.dtype)
  def test_shape_small_rect_n(self): verify_asm_gemm(1, 256, 512, 256, dtype=self.dtype)
  def test_shape_small_rect_k(self): verify_asm_gemm(1, 256, 256, 512, dtype=self.dtype)
  def test_shape_tall(self): verify_asm_gemm(1, 2048, 256, 256, dtype=self.dtype)
  def test_shape_wide(self): verify_asm_gemm(1, 256, 2048, 256, dtype=self.dtype)
  def test_shape_deep(self): verify_asm_gemm(1, 256, 256, 4096, dtype=self.dtype)
  def test_shape_non_square(self): verify_asm_gemm(1, 1024, 2048, 512, dtype=self.dtype)
  def test_shape_batched_small(self): verify_asm_gemm(2, 256, 256, 256, dtype=self.dtype)
  def test_shape_batched_rect(self): verify_asm_gemm(2, 512, 1024, 256, dtype=self.dtype)
  # K edge cases: change iters to exercise different loop paths, k big enough for hk kernel
  def test_shape_k256(self): verify_asm_gemm(1, 256, 256, 256, dtype=self.dtype)
  def test_shape_k512(self): verify_asm_gemm(1, 256, 256, 512, dtype=self.dtype)
  def test_shape_k768(self): verify_asm_gemm(1, 256, 256, 768, dtype=self.dtype)

  def test_llama3_out1(self): verify_asm_gemm(1, 8192, 128256, 4096, dtype=self.dtype)
  def test_llama3_out2(self): verify_asm_gemm(1, 8192, 4096, 128256, dtype=self.dtype)
  def test_llama3_out3(self): verify_asm_gemm(1, 4096, 128256, 8192, dtype=self.dtype)

# mxfp8: 1x32 block scaling along K, e8m0 scales packed iteration-major (K/128, dim) uint32
def quantize_mxfp8(x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  rows, K = x.shape
  scale_K, k_iters = K // 32, K // 128
  xb = x.reshape(rows, scale_K, 32).float()
  amax = xb.abs().max(axis=-1)
  e8 = (amax.log2().floor() + 127).clamp(0, 254)
  e8 = (amax == 0).where(Tensor.zeros_like(e8), e8).cast(dtypes.uint8)
  xq = (xb * (127.0 - e8.cast(dtypes.float32)).exp2().reshape(rows, scale_K, 1)).cast(FP8_DTYPE).reshape(rows, K)
  packed = e8.reshape(rows, k_iters, 4).bitcast(dtypes.uint32).reshape(rows, k_iters).permute(1, 0)
  return xq.contiguous(), e8, packed.contiguous()

def dequant_mxfp8(xq:Tensor, e8:Tensor) -> Tensor:
  rows, K = xq.shape
  scale = (e8.cast(dtypes.float32) - 127.0).exp2()
  return (xq.float().reshape(rows, K // 32, 32) * scale.reshape(rows, K // 32, 1)).reshape(rows, K)

def run_mxfp8_gemm(M:int, N:int, K:int) -> None:
  import functools
  from extra.gemm.cdna_asm_gemm import custom_hk_mxfp8_gemm
  Tensor.manual_seed(0)
  a = (Tensor.randn(M, K, dtype=dtypes.float) * 0.5).realize()
  b = (Tensor.randn(N, K, dtype=dtypes.float) * 0.5).realize()
  a_q, a_e8, a_si = quantize_mxfp8(a)
  b_q, b_e8, b_si = quantize_mxfp8(b)
  Tensor.realize(a_q, a_e8, a_si, b_q, b_e8, b_si)

  out = Tensor.invalids(1, M, N, dtype=dtypes.bfloat16, device=a.device)
  tst = out.custom_kernel(a_q.unsqueeze(0), b_q, a_si, b_si, fxn=functools.partial(custom_hk_mxfp8_gemm, dname=a.device))[0].squeeze(0)
  ref_mx = dequant_mxfp8(a_q, a_e8) @ dequant_mxfp8(b_q, b_e8).T
  ref = a @ b.T
  Tensor.realize(tst, ref_mx, ref)
  if a.device.startswith("NULL"): return
  err_mx = ((tst.float() - ref_mx).abs().mean() / ref_mx.abs().mean()).item()
  err = ((tst.float() - ref).abs().mean() / ref.abs().mean()).item()
  assert err_mx < 1e-2, f"kernel vs mxfp8 reference rel err {err_mx}"
  assert err < 6e-2, f"kernel vs fp32 rel err {err}"

def run_mx_gemm_bw(M:int, N:int, K:int, w_post:bool=False) -> None:
  Tensor.manual_seed(0)
  a_rand = (Tensor.randn(M, K, dtype=dtypes.float) * 0.5).cast(dtypes.bfloat16).realize()
  b_rand = (Tensor.randn(N, K, dtype=dtypes.float) * 0.5).cast(dtypes.bfloat16).realize()
  w_post_scale = (Tensor.rand(N, dtype=dtypes.float) + 0.5).realize() if w_post else None
  a, b, a_ref, b_ref = a_rand.clone(), b_rand.clone(), a_rand.clone(), b_rand.clone()
  tst = asm_gemm(a, b.T, mx=True, w_post_scale=w_post_scale)
  tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)
  a_grad, b_grad = a.grad.float().contiguous().realize(), b.grad.float().contiguous().realize()
  ref = a_ref.float() @ b_ref.float().T
  if w_post is not None and w_post_scale is not None: ref = ref * w_post_scale.reshape(1, -1)
  ref.sum().backward()
  ref_b_grad = b_ref.grad / w_post_scale.reshape(-1, 1) if w_post_scale is not None else b_ref.grad
  Tensor.realize(ref, a_ref.grad, b_ref.grad)
  if a.device.startswith("NULL"): return
  for name, t, r in [("fw", tst, ref), ("grad_a", a_grad, a_ref.grad), ("grad_b", b_grad, ref_b_grad)]:
    err = ((t.float() - r.float()).abs().mean() / (r.float().abs().mean() + 1e-8)).item()
    assert err < 6e-2, f"{name} rel err {err}"

def run_mx_gemm_multi(M:int, N:int, K:int, x_shard, w_shard, g_shard, gpus:int=2) -> None:
  Tensor.manual_seed(0)
  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus))
  x_r = (Tensor.randn(M, K, dtype=dtypes.float) * 0.5).cast(dtypes.bfloat16).realize()
  w_r = (Tensor.randn(N, K, dtype=dtypes.float) * 0.5).cast(dtypes.bfloat16).realize()
  def run(shard):
    x = (x_r.shard(devs, axis=x_shard) if shard else x_r.clone())
    w = (w_r.shard(devs, axis=w_shard) if shard else w_r.clone())
    out = asm_gemm(x, w.T, mx=True)
    gmul = Tensor.ones(M, N).cast(dtypes.bfloat16)
    (out.float() * (gmul.shard(devs, axis=g_shard) if shard else gmul).float()).sum().backward()
    Tensor.realize(out, x.grad, w.grad)
    to = (lambda t: t.to(Device.DEFAULT)) if shard else (lambda t: t)
    return to(out).float().numpy(), to(x.grad).float().numpy(), to(w.grad).float().numpy()
  ref = run(False)
  if Device.DEFAULT.startswith("NULL"): return
  got = run(True)
  for name, g, r in zip(("fw", "grad_x", "grad_w"), got, ref):
    err = ((abs(g - r)).mean() / (abs(r).mean() + 1e-8))
    assert err < 2e-2, f"{name} sharded vs single rel err {err}"

def run_mx_prequant(M:int, N:int, K:int) -> None:
  from extra.gemm.cdna_asm_gemm import quantize_mxfp8
  Tensor.manual_seed(0)
  x_rand = (Tensor.randn(M, K, dtype=dtypes.float) * 0.5).cast(dtypes.bfloat16).realize()
  w_rand = (Tensor.randn(N, K, dtype=dtypes.float) * 0.5).cast(dtypes.bfloat16).realize()
  x, w = x_rand.clone(), w_rand.clone()
  x_q, x_e8, x_si = quantize_mxfp8(x)
  w_q, w_e8, w_si = quantize_mxfp8(w)
  out = asm_gemm(x_q, w_q.T, mx=True, mx_scales=(x_si, x_e8, w_si, w_e8))
  out.sum().backward()
  Tensor.realize(out, x.grad, w.grad)
  if Device.DEFAULT.startswith("NULL"): return
  ref_out, gx = x_rand.float() @ w_rand.float().T, w_rand.float().sum(0)
  gw = x_rand.float().sum(0).reshape(1, K).expand(N, K)
  for name, t, r in [("fw", out, ref_out), ("grad_x", x.grad, gx), ("grad_w", w.grad, gw)]:
    err = ((t.float() - r.float()).abs().mean() / (r.float().abs().mean() + 1e-8)).item()
    assert err < 6e-2, f"{name} prequant vs analytic rel err {err}"

@unittest.skipUnless(has_hipcc(), "requires hipcc to compile")
class TestGemmMXFP8(unittest.TestCase):
  def setUp(self):
    if not is_cdna4() or DEV.interface.startswith("MOCK"): self.skipTest("mxfp8 gemm is only for cdna4")
  def test_prequant_simple(self): run_mx_prequant(256, 256, 256)
  def test_prequant_rect(self): run_mx_prequant(512, 256, 512)
  def test_simple(self): run_mxfp8_gemm(N:=getenv("N", 256), N, 2*128)
  def test_rect(self): run_mxfp8_gemm(512, 256, 512)
  def test_llama_ffn(self): run_mxfp8_gemm(8192, 14336, 4096)
  def test_llama_ffn2(self): run_mxfp8_gemm(8192, 4096, 14336)
  def test_llama_qkv(self): run_mxfp8_gemm(8192, 4096, 4096)
  def test_general_n_fw(self):
    for N in (256, 1792, 2048, 8192): run_mxfp8_gemm(8192, N, 4096)
  # backward needs all dims tile-aligned (dgrad reduces N, wgrad reduces M)
  def test_bw_simple(self): run_mx_gemm_bw(256, 256, 256)
  def test_bw_rect(self): run_mx_gemm_bw(512, 256, 512)
  def test_bw_w_post(self): run_mx_gemm_bw(256, 256, 256, w_post=True)
  def test_bw_llama_qkv(self): run_mx_gemm_bw(8192, 4096, 4096)
  def test_general_n_bw(self):
    for N in (2048, 8192, 14336): run_mx_gemm_bw(8192, N, 4096)
  # MP sharding: col-parallel (w on out axis), row-parallel (x,w on in axis)
  @needs_second_gpu
  def test_multi_col_parallel(self): run_mx_gemm_multi(512, 512, 512, x_shard=None, w_shard=0, g_shard=1)
  @needs_second_gpu
  def test_multi_row_parallel(self): run_mx_gemm_multi(512, 512, 512, x_shard=1, w_shard=1, g_shard=None)
  @needs_second_gpu
  def test_multi_data_parallel(self): run_mx_gemm_multi(512, 512, 512, x_shard=0, w_shard=None, g_shard=0)

def run_atb_gemm(rows, M, N, a_shard=None, b_shard=None, gpus=1, atol=1.0, rtol=3e-2) -> None:
  import numpy as np
  Tensor.manual_seed(0)
  a = Tensor.randn(1, rows, M, dtype=dtypes.float).cast(dtypes.bfloat16)
  b = Tensor.randn(1, rows, N, dtype=dtypes.float).cast(dtypes.bfloat16)
  with Context(DEBUG=0): Tensor.realize(a, b)
  ref = (a[0].float().transpose(0, 1) @ b[0].float()).realize()  # [M, N]
  if gpus > 1:
    devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus))
    a, b = a.shard(devs, axis=a_shard), b.shard(devs, axis=b_shard)
  out = hk_bf16_atb_gemm(a, b)
  np.testing.assert_allclose(out.float().numpy(), ref.numpy(), atol=atol, rtol=rtol)

@unittest.skipUnless(has_hipcc(), "requires hipcc to compile")
class TestHkBf16AtbGemm(unittest.TestCase):
  def setUp(self):
    if not is_cdna4(): self.skipTest("hk bf16 atb gemm is cdna4 only")
  def test_single(self): run_atb_gemm(256, 256, 256)
  @needs_second_gpu
  def test_k_sharded(self): run_atb_gemm(512, 256, 256, a_shard=1, b_shard=1, gpus=2)
  @needs_second_gpu
  def test_n_sharded(self): run_atb_gemm(256, 256, 512, a_shard=None, b_shard=2, gpus=2)
  @needs_second_gpu
  def test_m_sharded(self): run_atb_gemm(256, 512, 256, a_shard=2, b_shard=None, gpus=2)

if __name__ == "__main__":
  unittest.main()
