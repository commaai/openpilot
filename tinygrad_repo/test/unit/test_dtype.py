import unittest
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, DType, ImageDType, PtrDType, to_dtype

class TestImageDType(unittest.TestCase):
  def test_image_scalar(self):
    assert dtypes.imagef((10,10)).base.scalar() == dtypes.float32
    assert dtypes.imageh((10,10)).base.scalar() == dtypes.float32
  def test_image_vec(self):
    assert dtypes.imagef((10,10)).base.vec(4) == dtypes.float32.vec(4)
    assert dtypes.imageh((10,10)).base.vec(4) == dtypes.float32.vec(4)

class TestEqStrDType(unittest.TestCase):
  def test_image_ne(self):
    if ImageDType is None: raise unittest.SkipTest("no ImageDType support")
    assert dtypes.float == dtypes.float32, "float doesn't match?"
    assert dtypes.imagef((1,2,4)) != dtypes.imageh((1,2,4)), "different image dtype doesn't match"
    assert dtypes.imageh((1,2,4)) != dtypes.imageh((1,4,2)), "different shape doesn't match"
    assert dtypes.imageh((1,2,4)) == dtypes.imageh((1,2,4)), "same shape matches"
    assert isinstance(dtypes.imageh((1,2,4)), ImageDType)
  def test_ptr_eq(self):
    assert dtypes.float32.ptr() == dtypes.float32.ptr()
    assert not (dtypes.float32.ptr() != dtypes.float32.ptr())
  def test_strs(self):
    if PtrDType is None: raise unittest.SkipTest("no PtrDType support")
    self.assertEqual(str(dtypes.imagef((1,2,4))), "dtypes.imagef((1, 2, 4))")
    self.assertEqual(str(dtypes.float32.ptr(16)), "dtypes.float.ptr(16)")

class TestToDtype(unittest.TestCase):
  def test_dtype_to_dtype(self):
    dtype = dtypes.int32
    res = to_dtype(dtype)
    self.assertIsInstance(res, DType)
    self.assertEqual(res, dtypes.int32)

  def test_str_to_dtype(self):
    dtype = "int32"
    res = to_dtype(dtype)
    self.assertIsInstance(res, DType)
    self.assertEqual(res, dtypes.int32)

class TestCastConvenienceMethod(unittest.TestCase):
  def test_method(self):
    for input_dtype in (dtypes.float, dtypes.int):
      t = Tensor([1, 2], dtype=input_dtype)
      self.assertEqual(t.dtype, input_dtype)
      self.assertEqual(t.bool().dtype, dtypes.bool)
      self.assertEqual(t.short().dtype, dtypes.short)
      self.assertEqual(t.int().dtype, dtypes.int)
      self.assertEqual(t.long().dtype, dtypes.long)
      self.assertEqual(t.half().dtype, dtypes.half)
      self.assertEqual(t.bfloat16().dtype, dtypes.bfloat16)
      self.assertEqual(t.float().dtype, dtypes.float)
      self.assertEqual(t.double().dtype, dtypes.double)

if __name__ == "__main__":
  unittest.main()
