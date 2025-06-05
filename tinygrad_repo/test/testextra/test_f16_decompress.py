import unittest
from extra.f16_decompress import u32_to_f16
from tinygrad.tensor import Tensor
from tinygrad.device import is_dtype_supported
from tinygrad import dtypes
import numpy as np

class TestF16Decompression(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16")
  def test_u32_to_f16(self):
    a = Tensor.randn(50, dtype=dtypes.float16)
    f16_as_u32 = a.bitcast(dtypes.uint32)
    f16 = u32_to_f16(f16_as_u32)
    ref = a.numpy()
    out = f16.numpy().astype(np.float16)
    np.testing.assert_allclose(out, ref)
