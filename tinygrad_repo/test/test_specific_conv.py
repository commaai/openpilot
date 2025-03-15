import unittest
from tinygrad.helpers import CI
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import is_dtype_supported
# similar to test/external/external_test_gpu_ast.py, but universal

@unittest.skipIf(Device.DEFAULT in {"CUDA", "NV"} and CI, "slow on CUDA CI")
class TestSpecific(unittest.TestCase):
  # from openpilot

  # 1x1 6 <- 24
  def test_1x1_6_24(self):
    x = Tensor.randn(1,   24*4, 32, 64)
    w = Tensor.randn(6*4, 24*4, 1,  1)
    x.conv2d(w).permute(0,2,3,1).reshape(32, 384, 4).contiguous().realize()

  def test_vec_mul(self):
    # this forces it to be an image...
    x = Tensor.ones(1, 512, 4).contiguous().reshape(1, 2048)
    w = Tensor.randn(2048, 512)
    (x @ w).reshape(1, 128, 4).contiguous().realize()

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16 support")
  def test_big_vec_mul(self):
    # from LLaMA
    #   0 buffer<4096, dtypes.float>                      [View((1024, 1, 1, 4), (4, 0, 0, 1), 0, None)]
    #   1 buffer<4096, dtypes.float>                      [View((1024, 1024, 4, 4), (0, 4, 1, 0), 0, None)]
    #   2 buffer<16777216, dtypes.half>                   [View((1024, 1024, 4, 4), (16384, 4, 1, 4096), 0, None)]
    x = Tensor.randn(4096).realize()
    w = Tensor.randn(4096, 4096, dtype=dtypes.float16).realize()
    (x @ w.T).realize()

  # from https://dl.acm.org/doi/pdf/10.1145/3495243.3517020

  # ~260 GFLOPS on Adreno 640, should be 260*(720/890)*(596/710) = 176.5 on downclocked 630
  # we get 170
  def test_1x1_28_28(self):
    x = Tensor.randn(1,   256, 28, 28)
    w = Tensor.randn(256, 256, 1,  1)
    x.conv2d(w).permute(0,2,3,1).reshape(28, 28*256//4, 4).contiguous().realize()

  # 132 GFLOPS on Adreno 640, should be 132*(720/890)*(596/710) = 90 on downclocked 630
  # gets 54 with broken opt, 74 without opt, and 146 if we pad and opt 3!
  def test_3x3_28_28_stride_2(self):
    x = Tensor.randn(1,   288, 36, 36)
    w = Tensor.randn(384, 288, 3,  3)
    x.conv2d(w, stride=2).permute(0,2,3,1).reshape(17, 17*384//4, 4).contiguous().realize()

  def test_3x3_28_28_stride_2_padded(self):
    x = Tensor.randn(1,   288, 36, 36)
    w = Tensor.randn(384, 288, 3,  3)
    x.conv2d(w, stride=2, padding=1).permute(0,2,3,1).reshape(18, 18*384//4, 4).contiguous().realize()

if __name__ == '__main__':
  unittest.main()
