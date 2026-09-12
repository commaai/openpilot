import unittest, pickle
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes, DType, to_dtype, Invalid, InvalidType

class TestEqStrDType(unittest.TestCase):
  def test_strs(self):
    self.assertEqual(str(dtypes.float32), "dtypes.float")

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

class TestDtypeTolist(unittest.TestCase):
  def test_bfloat16(self):
    self.assertEqual(Tensor([-60000, 1.5, 3.1, 60000], device="PYTHON", dtype=dtypes.bfloat16).tolist(), [-59904.0, 1.5, 3.09375, 59904.0])
  def test_fp8(self):
    # 448
    self.assertEqual(Tensor([-30000, 1.5, 3.1, 30000], device="PYTHON", dtype=dtypes.fp8e4m3).tolist(), [-448.0, 1.5, 3.0, 448.0])
    # 57344
    self.assertEqual(Tensor([-30000, 1.5, 3.1, 30000], device="PYTHON", dtype=dtypes.fp8e5m2).tolist(), [-28672.0, 1.5, 3.0, 28672.0])

class TestCanLosslessCast(unittest.TestCase):
  def test_can_lossless_cast(self):
    from tinygrad.dtype import can_lossless_cast
    # signed -> unsigned is NOT lossless (negative values wrap)
    self.assertFalse(can_lossless_cast(dtypes.int8, dtypes.uint64))
    self.assertFalse(can_lossless_cast(dtypes.int32, dtypes.uint32))
    # unsigned -> larger signed is lossless
    self.assertTrue(can_lossless_cast(dtypes.uint8, dtypes.int16))
    self.assertTrue(can_lossless_cast(dtypes.uint32, dtypes.int64))
    # large ints don't fit in floats
    self.assertFalse(can_lossless_cast(dtypes.int32, dtypes.float))
    self.assertFalse(can_lossless_cast(dtypes.int64, dtypes.double))
    # half has more mantissa bits
    self.assertTrue(can_lossless_cast(dtypes.int8, dtypes.half))
    self.assertFalse(can_lossless_cast(dtypes.int8, dtypes.bfloat16))

class TestInvalidSingleton(unittest.TestCase):
  def test_singleton(self):
    self.assertIs(InvalidType(), InvalidType())
    self.assertIs(InvalidType(), Invalid)
  def test_pickle(self):
    self.assertIs(pickle.loads(pickle.dumps(Invalid)), Invalid)

if __name__ == "__main__":
  unittest.main()
