import unittest
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.helpers import getenv
from extra.gemm.asm.cdna.gemm import asm_gemm

def verify_asm_gemm(batch:int, M:int, N:int, K:int, dtype=dtypes.bfloat16, gpus:int=1) -> None:
  Tensor.manual_seed(0)
  a_rand = Tensor.randn((batch, M, K), dtype=dtypes.float).sub(0.5).cast(dtype)
  b_rand = Tensor.randn((K, N), dtype=dtypes.float).sub(0.5).cast(dtype)
  with Context(DEBUG=0):
    Tensor.realize(a_rand, b_rand)

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(gpus)) if (multi:=gpus>1) else None

  a, b = Tensor(a_rand.numpy(), requires_grad=True).cast(dtype), Tensor(b_rand.numpy(), requires_grad=True).cast(dtype)
  if multi: a, b = a.shard(devs, axis=0), b.shard(devs, axis=None)
  tst = asm_gemm(a, b)
  tst.sum().backward()
  Tensor.realize(tst, a.grad, b.grad)

  a_ref, b_ref = Tensor(a_rand.numpy(), requires_grad=True).cast(dtype), Tensor(b_rand.numpy(), requires_grad=True).cast(dtype)
  if multi: a_ref, b_ref = a_ref.shard(devs, axis=0), b_ref.shard(devs, axis=None)
  with Context(ASM_GEMM=0): ref = a_ref @ b_ref
  ref.sum().backward()
  Tensor.realize(ref, a_ref.grad, b_ref.grad)

  with Context(DEBUG=0):
    assert (tst - ref).square().max().float().item() < 1e-6, "forward mismatch"
    assert (a.grad - a_ref.grad).square().max().float().item() < 1e-3, "grad_a mismatch"
    assert (b.grad - b_ref.grad).square().max().float().item() < 1e-3, "grad_b mismatch"

class TestGemm(unittest.TestCase):
  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 4096), N, N, dtype=dtypes.half)
  def test_gemm(self): verify_asm_gemm(1, 8192, 4096, 14336)
  def test_gemm_multi(self): verify_asm_gemm(2, 8192, 4096, 14336, gpus=2)
  def test_gemm_unsupported(self):
    with self.assertRaisesRegex(AssertionError, "shape not supported"):
      verify_asm_gemm(8, 8192, 1024, 4096, gpus=8)

class TestGemmLarge(unittest.TestCase):
  def setUp(self):
    if getattr(Device[Device.DEFAULT].renderer, "arch", "") != "gfx950":
      self.skipTest("very slow on non mi350x")

  def test_gemm1(self): verify_asm_gemm(8, 8192, 4096, 14336, gpus=8)
  def test_gemm2(self): verify_asm_gemm(8, 8192, 128256, 4096, gpus=8)
  def test_gemm3(self): verify_asm_gemm(8, 8192, 14336, 4096, gpus=8)
  def test_gemm4(self): verify_asm_gemm(8, 4096, 14336, 4096, gpus=8)
  def test_gemm5(self): verify_asm_gemm(8, 4096, 4096, 14336, gpus=8)
  def test_gemm6(self): verify_asm_gemm(16, 4096, 4096, 14336, gpus=8)

if __name__ == "__main__":
  unittest.main()
