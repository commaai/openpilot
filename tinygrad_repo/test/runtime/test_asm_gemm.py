import unittest
from tinygrad import Device, dtypes
from tinygrad.helpers import getenv
from test.helpers import needs_second_gpu
from test.device.amd.test_asm_gemm import (is_cdna4, verify_asm_gemm, verify_asm_gemm_k_sharded, verify_asm_gemm_m_sharded,
                                         verify_asm_gemm_n_sharded, verify_asm_gemm_n_sharded_2d, verify_asm_gemm_k_sharded_3d)

# 128x smaller than usual
# uses the UOp GEMM, runs on non CDNA4 and CI
@unittest.skipUnless(dtypes.bfloat16 in Device[Device.DEFAULT].renderer.supported_dtypes(), "need half")
class TestGemm(unittest.TestCase):
  def setUp(self):
    if is_cdna4(): self.skipTest("shapes are too small for the assembly GEMM")
  def test_simple(self): verify_asm_gemm(1, N:=getenv("N", 32), N, N, dtype=dtypes.bfloat16)
  def test_gemm(self): verify_asm_gemm(1, 32, 32, 32)
  def test_gemm_batched(self): verify_asm_gemm(2, 32, 32, 32)
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

if __name__ == "__main__": unittest.main()
