import unittest
from dataclasses import replace
from itertools import islice
from tinygrad import Tensor, Device
from tinygrad.codegen import to_program
from tinygrad.engine.realize import time_call
from tinygrad.helpers import Context, DEBUG
from tinygrad.nn import Conv2d
from tinygrad.nn.state import get_parameters

class TestKernelSpeed(unittest.TestCase):
  def _get_tensor(self, *shape:int):
    with Context(BEAM=0, DEBUG=0):
      # TODO: randn is 20% faster than rand for gemv
      return Tensor.randn(shape, dtype="half").realize()

  def _time_kernel(self, out:Tensor, beam:int):
    linear = out.schedule_linear()
    self.assertEqual(len(linear.src), 1, "expected a single kernel")
    call = linear.src[0]
    prg = to_program(call.src[0].replace(arg=replace(call.src[0].arg, beam=beam)), Device[out.device].renderer)
    return min(islice(time_call(call.replace(src=(prg, *call.src[1:])), clear_l2=True), 3, 10))

  def _compare(self, tm, tflops, gbs, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
      if DEBUG >= 1:
        print(f"{tm=:.6f}")
        print(f"{tflops=:.6f}")
        print(f"{gbs=:.3f}")

      if Device.DEFAULT == "NV":
        if nv_tflops is not None:
          if DEBUG >=1: print(f"tflop/s target: {nv_tflops}")
          self.assertGreater(tflops, nv_tflops)
        if nv_gbs is not None:
          if DEBUG >=1: print(f"gb/s target: {nv_gbs}")
          self.assertGreater(gbs, nv_gbs)

      if Device.DEFAULT == "AMD":
        if amd_tflops is not None:
          if DEBUG >=1: print(f"tflop/s target: {amd_tflops}")
          self.assertGreater(tflops, amd_tflops)
        if amd_gbs is not None:
          if DEBUG >=1: print(f"gb/s target: {amd_gbs}")
          self.assertGreater(gbs, amd_gbs)

  def _test_matmul(self, M, K=None, N=None, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
    # (MxK) @ (KxN)
    if N is None: N = M
    if K is None: K = M
    a = self._get_tensor(M, K)
    b = self._get_tensor(K, N)
    tm = self._time_kernel(c:=a @ b, beam=3)

    ops = 2 * M * N * K
    mems = a.dtype.itemsize * M * K + b.dtype.itemsize * K * N + c.dtype.itemsize * M * N
    tflops = ops / tm / 1e12
    gbs = mems / tm / 1e9
    self._compare(tm, tflops, gbs, nv_tflops, nv_gbs, amd_tflops, amd_gbs)

  def _test_conv_3x3(self, BS, CIN, COUT, H, W, nv_tflops=None, nv_gbs=None, amd_tflops=None, amd_gbs=None):
    K = 3
    with Context(BEAM=0, DEBUG=0):
      conv = Conv2d(CIN, COUT, K, padding=1)
      Tensor.realize(*get_parameters(conv))

    x = self._get_tensor(BS, CIN, H, W)
    tm = self._time_kernel(_c:=conv(x), beam=2)

    # naive algo
    ops = 2 * BS * CIN * COUT * K * K * H * W
    mems = x.nbytes() + conv.weight.nbytes() + conv.bias.nbytes() + _c.nbytes()
    tflops = ops / tm / 1e12
    gbs = mems / tm / 1e9
    self._compare(tm, tflops, gbs, nv_tflops, nv_gbs, amd_tflops, amd_gbs)

  # TODO: why are convs so slow?!?
  def test_conv_3x3_256_32_32_256_256(self): self._test_conv_3x3(256, 32, 32, 256, 256, nv_tflops=27, amd_tflops=13)

  # theoretical is nv_tflops=165, amd_tflops=123
  def test_gemm_4096(self): self._test_matmul(4096, nv_tflops=109, amd_tflops=65)
  def test_gemm_8192(self): self._test_matmul(8192, nv_tflops=115, amd_tflops=60)

  # theoretical is nv_gbs=1008, amd_gbs=960
  def test_gemv_16384_4096(self): self._test_matmul(16384, 4096, 1, nv_gbs=840, amd_gbs=750)
  def test_gemv_4096_16384(self): self._test_matmul(4096, 16384, 1, nv_gbs=820, amd_gbs=750)

if __name__ == '__main__':
  unittest.main()
