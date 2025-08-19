import unittest, math
import numpy as np
import torch
from typing import Any, List
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import getenv, DEBUG, CI
from tinygrad.dtype import DType, DTYPES_DICT, least_upper_dtype, fp8_to_float, float_to_fp8
from tinygrad import Device, Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
from hypothesis import assume, given, settings, strategies as strat
from test.helpers import rand_for_dtype
from test.unit.test_dtype_spec import _assert_eq, core_dtypes, dtype_ints, dtype_floats, FP8E4M3_MAX, FP8E5M2_MAX
import ml_dtypes
import pytest
pytestmark = pytest.mark.filterwarnings("ignore")

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

if Device.DEFAULT == "CPU": core_dtypes.remove(dtypes.bfloat16)  # NOTE: this is for teenygrad, don't remove

def get_available_cast_dtypes(dtype: DType) -> List[DType]:
  if not is_dtype_supported(dtype): return []
  # dont cast internal dtypes
  return [v for k, v in DTYPES_DICT.items() if v != dtype and is_dtype_supported(v) and not k.startswith("_")]

def _test_to_np(a:Tensor, np_dtype, target):
  if DEBUG >= 2: print(a)
  na = a.numpy()
  if DEBUG >= 2: print(na, na.dtype, a.uop.base.realized)
  try:
    assert na.dtype == np_dtype
    np.testing.assert_allclose(na, target)
  except AssertionError as e:
    raise AssertionError(f"\ntensor {a.numpy()} does not match target {target} with np_dtype {np_dtype}") from e

def _test_op(fxn, target_dtype:DType, target):
  _assert_eq(fxn(), target_dtype, target)
def _test_cast(a:Tensor, target_dtype:DType):
  if a.is_floating_point() and dtypes.is_unsigned(target_dtype):
    # converting negative float to unsigned integer is undefined
    a = a.abs()
  if target_dtype == dtypes.half and Device.DEFAULT == "PYTHON":
    # TODO: struct.pack cannot pack value > 65504 (max of half) into e format
    a = (a > 65504).where(65504, a)

  _test_op(lambda: a.cast(target_dtype), target_dtype, list(a.numpy().astype(_to_np_dtype(target_dtype))))
def _test_bitcast(a:Tensor, target_dtype:DType, target=None):
  if target_dtype == dtypes.bfloat16: raise unittest.SkipTest("no test for bf16 bitcast yet")
  if getenv("PTX") and a.dtype == dtypes.int8 and target_dtype.itemsize != a.dtype.itemsize:
    raise unittest.SkipTest("shape changing bitcast of int8 broken on PTX")
  _test_op(lambda: a.bitcast(target_dtype), target_dtype, target or a.numpy().view(_to_np_dtype(target_dtype)).tolist())

class TestDType(unittest.TestCase):
  DTYPE: Any = None
  DATA: Any = None
  @classmethod
  def setUpClass(cls):
    if not cls.DTYPE or not is_dtype_supported(cls.DTYPE): raise unittest.SkipTest("dtype not supported")
    cls.DATA = rand_for_dtype(cls.DTYPE, 10)
  def setUp(self):
    if self.DTYPE is None: raise unittest.SkipTest("base class")

  def test_to_np(self):
    _test_to_np(Tensor(self.DATA, dtype=self.DTYPE), _to_np_dtype(self.DTYPE), np.array(self.DATA, dtype=_to_np_dtype(self.DTYPE)))

  def test_casts_to(self): list(map(
    lambda dtype: _test_cast(Tensor(self.DATA, dtype=dtype), self.DTYPE),
    get_available_cast_dtypes(self.DTYPE)
  ))
  def test_casts_from(self): list(map(
    lambda dtype: _test_cast(Tensor(self.DATA, dtype=self.DTYPE), dtype),
    get_available_cast_dtypes(self.DTYPE)
  ))

  def test_same_size_ops(self):
    list(map(
      lambda dtype: _test_ops(a_dtype=self.DTYPE, b_dtype=dtype) if dtype.itemsize == self.DTYPE.itemsize else None,
      get_available_cast_dtypes(self.DTYPE)
    ))
  def test_upcast_ops(self):
    list(map(
      lambda dtype: _test_ops(a_dtype=self.DTYPE, b_dtype=dtype) if dtype.itemsize > self.DTYPE.itemsize else None,
      get_available_cast_dtypes(self.DTYPE)
  ))
  def test_upcast_to_ops(self):
    list(map(
      lambda dtype: _test_ops(a_dtype=dtype, b_dtype=self.DTYPE) if dtype.itemsize < self.DTYPE.itemsize else None,
      get_available_cast_dtypes(self.DTYPE)
  ))
  def test_bitcast(self):
    if self.DTYPE == dtypes.bool: raise unittest.SkipTest("no bools in bitcast")
    list(map(
      lambda dtype:
        _test_bitcast(Tensor(self.DATA[:8], dtype=self.DTYPE), dtype) if dtype != dtypes.bool else None,
     get_available_cast_dtypes(self.DTYPE)
    ))

  @unittest.skipIf(Device.DEFAULT == "PYTHON", "skip for now")
  @unittest.skipIf(getenv("PTX"), "skip for now")
  def test_uint_overflow(self):
    if not dtypes.is_unsigned(self.DTYPE): raise unittest.SkipTest("only for unsigned")
    v = dtypes.max(self.DTYPE)
    _test_to_np(Tensor(v, dtype=self.DTYPE)+2, _to_np_dtype(self.DTYPE), np.array(v, dtype=_to_np_dtype(self.DTYPE))+2)
    _test_to_np(Tensor(v, dtype=self.DTYPE)*2, _to_np_dtype(self.DTYPE), np.array(v, dtype=_to_np_dtype(self.DTYPE))*2)

  def test_dtypes_fields(self):
    fields = dtypes.fields()
    self.assertIn("float", fields)
    self.assertIn("float32", fields)
    self.assertEqual(len(fields), 26)
    self.assertTrue(all(isinstance(value, DType) for value in fields.values()))
    self.assertTrue(all(issubclass(_to_np_dtype(value), np.generic) for value in fields.values() if _to_np_dtype(value) is not None))

  def test_resulting_and_init_dtypes_match(self):
    dtypes = list(map(np.dtype, ["bool", "uint8", "int8", "int16", "int32", "int64", "float32", "float64"]))
    data = [1., 2., 0., 0.5, -1.5, 5.25]
    for dt in dtypes:
      arr = np.asarray(data).astype(dt)
      tensor = Tensor(arr)
      if not is_dtype_supported(tensor.dtype): continue
      tin = tensor.numpy()
      tor = torch.as_tensor(arr).detach().numpy()
      assert dt == tin.dtype == tor.dtype, f"dtype mismatch: expected={dt} | tinygrad={tin.dtype} | torch={tor.dtype}"
      np.testing.assert_allclose(tin, tor, atol=1e-6, rtol=1e-3)

  def test_finfo(self):
    if self.DTYPE not in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]: return
    info = np.finfo(_to_np_dtype(self.DTYPE))
    assert info.bits == self.DTYPE.itemsize*8
    assert info.nexp == dtypes.finfo(self.DTYPE)[0]
    assert info.nmant == dtypes.finfo(self.DTYPE)[1]

def _test_ops(a_dtype:DType, b_dtype:DType, target_dtype=None):
  target_dtype = target_dtype or least_upper_dtype(a_dtype, b_dtype)
  if not is_dtype_supported(a_dtype) or not is_dtype_supported(b_dtype) or not is_dtype_supported(target_dtype): return
  if a_dtype == dtypes.bool or b_dtype == dtypes.bool: return
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)+Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [2,4,6,8])
  _assert_eq((Tensor([1], dtype=a_dtype).cast(b_dtype)+Tensor([1], dtype=a_dtype).cast(b_dtype)).cast(a_dtype), a_dtype, [2])
  _assert_eq(Tensor([1,2,3,4], dtype=a_dtype)*Tensor([1,2,3,4], dtype=b_dtype), target_dtype, [1,4,9,16])
  _assert_eq(Tensor([[1,2],[3,4]], dtype=a_dtype)@Tensor.eye(2, dtype=b_dtype), target_dtype, [[1,2],[3,4]])
  _assert_eq(Tensor([1,1,1,1], dtype=a_dtype)+Tensor.ones((4,4), dtype=b_dtype), target_dtype, 2*Tensor.ones(4,4).numpy())

class TestFp8s(unittest.TestCase):
  def test_fp8e4m3_creation(self): assert Tensor([-1, 1, 2], dtype=dtypes.fp8e4m3).dtype == dtypes.fp8e4m3
  def test_fp8e5m2_creation(self): assert Tensor([-1, 1, 2], dtype=dtypes.fp8e5m2).dtype == dtypes.fp8e5m2

class TestFp8sConversions(unittest.TestCase):
  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=False, allow_infinity=False, min_value=-FP8E4M3_MAX, max_value=FP8E4M3_MAX))
  def test_float_to_fp8e4m3(self, x): np.testing.assert_equal(float_to_fp8(x, dtypes.fp8e4m3), ml_dtypes.float8_e4m3fn(x).tobytes()[0])

  def test_float_to_fp8e4m3_extreme_values(self):
    np.testing.assert_equal(float_to_fp8(FP8E4M3_MAX, dtypes.fp8e4m3), 126)
    np.testing.assert_equal(float_to_fp8(FP8E4M3_MAX*1.01, dtypes.fp8e4m3), 126)
    np.testing.assert_equal(float_to_fp8(math.inf, dtypes.fp8e4m3), 126)
    np.testing.assert_equal(float_to_fp8(-FP8E4M3_MAX, dtypes.fp8e4m3), 254)
    np.testing.assert_equal(float_to_fp8(-FP8E4M3_MAX*1.01, dtypes.fp8e4m3), 254)
    np.testing.assert_equal(float_to_fp8(-math.inf, dtypes.fp8e4m3), 254)
    np.testing.assert_equal(float_to_fp8(math.nan, dtypes.fp8e4m3), 127)
    np.testing.assert_equal(float_to_fp8(-math.nan, dtypes.fp8e4m3), 255)

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=False, allow_infinity=False, min_value=-FP8E5M2_MAX, max_value=FP8E5M2_MAX))
  def test_float_to_fp8e5m2(self, x): np.testing.assert_equal(float_to_fp8(x, dtypes.fp8e5m2), ml_dtypes.float8_e5m2(x).tobytes()[0])

  def test_float_to_fp8e5m2_extreme_values(self):
    np.testing.assert_equal(float_to_fp8(FP8E5M2_MAX, dtypes.fp8e5m2), 123)
    np.testing.assert_equal(float_to_fp8(FP8E5M2_MAX*1.01, dtypes.fp8e5m2), 123)
    np.testing.assert_equal(float_to_fp8(math.inf, dtypes.fp8e5m2), 123)
    np.testing.assert_equal(float_to_fp8(-FP8E5M2_MAX, dtypes.fp8e5m2), 251)
    np.testing.assert_equal(float_to_fp8(-FP8E5M2_MAX*1.01, dtypes.fp8e5m2), 251)
    np.testing.assert_equal(float_to_fp8(-math.inf, dtypes.fp8e5m2), 251)
    np.testing.assert_equal(float_to_fp8(math.nan, dtypes.fp8e5m2), 126)
    np.testing.assert_equal(float_to_fp8(-math.nan, dtypes.fp8e5m2), 254)

  @given(strat.integers(min_value=0, max_value=255))
  def test_fp8e4m3_to_float(self, x): np.testing.assert_equal(fp8_to_float(x, dtypes.fp8e4m3), np.uint8(x).view(ml_dtypes.float8_e4m3fn).item())

  @given(strat.integers(min_value=0, max_value=255))
  def test_fp8e5m2_to_float(self, x): np.testing.assert_equal(fp8_to_float(x, dtypes.fp8e5m2), np.uint8(x).view(ml_dtypes.float8_e5m2).item())

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "bfloat16 not supported")
class TestBFloat16(unittest.TestCase):
  def test_bf16_creation_numpy(self):
    data = [-1, 1, 2]
    t = Tensor(data, dtype=dtypes.bfloat16)
    assert t.dtype == dtypes.bfloat16
    tnp = t.numpy()
    assert tnp.dtype == np.float32
    np.testing.assert_allclose(tnp, np.array(data))

  def test_bf16_ones(self):
    t = Tensor.ones(3, 5, dtype=dtypes.bfloat16)
    assert t.dtype == dtypes.bfloat16
    np.testing.assert_allclose(t.numpy(), np.ones((3, 5)))

  def test_bf16_eye(self):
    t = Tensor.eye(3, dtype=dtypes.bfloat16)
    assert t.dtype == dtypes.bfloat16
    np.testing.assert_allclose(t.numpy(), np.eye(3))

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "bfloat16 not supported")
class TestBFloat16DType(unittest.TestCase):
  def test_bf16_to_float(self):
    _test_cast(Tensor([100000], dtype=dtypes.bfloat16), dtypes.float32)

  def test_float_to_bf16(self):
    _test_cast(Tensor([100000], dtype=dtypes.float32), dtypes.bfloat16)

  def test_bf16(self):
    t = Tensor([10000, -1, -1000, -10000, 20]).cast(dtypes.bfloat16)
    t.realize()
    back = t.cast(dtypes.float32)
    assert tuple(back.numpy().tolist()) == (9984., -1, -1000, -9984, 20)

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "bfloat16 not supported")
class TestBFloat16DTypeCast(unittest.TestCase):
  def test_f16_to_bf16_conversion(self):
    original_tensor = Tensor([1.0, 2.0, 3.0], dtype=dtypes.float16)
    converted_tensor = original_tensor.cast(dtypes.bfloat16)
    self.assertEqual(converted_tensor.dtype, dtypes.bfloat16)
    back_to_float32 = converted_tensor.cast(dtypes.float32)
    original_to_float32 = original_tensor.cast(dtypes.float32)
    np.testing.assert_allclose(back_to_float32.numpy(), original_to_float32.numpy(), rtol=1e-2, atol=1e-3)

  def test_f16_to_bf16_edge_cases(self):
    edge_cases = Tensor([0.0, -0.0, float('inf'), float('-inf'), float('nan')], dtype=dtypes.float16)
    converted = edge_cases.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_equal(converted.numpy(), edge_cases.cast(dtypes.float32).numpy())

  def test_f16_to_bf16_range_precision(self):
    large_value = Tensor([65504.0], dtype=dtypes.float16)  # Max representable in float16
    small_value = Tensor([6.1035e-5], dtype=dtypes.float16)  # Smallest positive normal float16
    large_converted = large_value.cast(dtypes.bfloat16).cast(dtypes.float32)
    small_converted = small_value.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_allclose(large_converted.numpy(), large_value.cast(dtypes.float32).numpy(), rtol=1e-2, atol=1e-3)
    np.testing.assert_equal(small_converted.numpy(), small_value.cast(dtypes.float32).numpy())

  def test_f16_to_bf16_randomized(self):
    np.random.seed(42)  # For reproducibility
    random_values = Tensor(np.random.uniform(-65504, 65504, 1000), dtype=dtypes.float16)
    converted = random_values.cast(dtypes.bfloat16).cast(dtypes.float32)
    np.testing.assert_allclose(converted.numpy(), random_values.cast(dtypes.float32).numpy(), rtol=1e-2, atol=1e-3)

class TestHalfDType(TestDType): DTYPE = dtypes.half

class TestFloatDType(TestDType):
  DTYPE = dtypes.float

  def test_float_to_uint(self):
    _test_op(lambda: Tensor([-0.9, -0.3, 1.2], dtype=dtypes.float32).cast(dtypes.uint32), dtypes.uint32,
             [0, 0, 1])

class TestDoubleDType(TestDType):
  DTYPE = dtypes.double
  @unittest.skipIf((CI and Device.DEFAULT in {"CUDA", "NV"}) or getenv("PTX"), "conversion not supported on CI CUDA and PTX")  # TODO: why not?
  def test_float64_increased_precision(self):
    for func in [
      lambda t: t.exp(),
      lambda t: t.exp2(),
      lambda t: t.log(),
      lambda t: t.log2(),
      lambda t: t.sqrt(),
      lambda t: t.rsqrt(),
      lambda t: t.sin(),
      lambda t: t.cos(),
      lambda t: t.tan(),
      lambda t: t.sigmoid(),
    ]:
      a = [2, 3, 4]
      np.testing.assert_allclose(func(Tensor(a, dtype=self.DTYPE)).numpy(), func(torch.tensor(a, dtype=torch.float64)), rtol=1e-12, atol=1e-12)

  def test_float64_to_float32_cast_inf(self):
    _test_op(lambda: Tensor([3.4e40, 3.4e38, 1, 0], dtype=dtypes.float64).cast(dtypes.float32),
             dtypes.float32, [float('inf'), 3.4e38, 1, 0])


class TestInt8DType(TestDType):
  DTYPE = dtypes.int8
  @unittest.skipIf(getenv("CUDA",0)==1 or getenv("PTX", 0)==1, "cuda saturation works differently")
  def test_int8_to_uint8_negative(self):
    _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint8), dtypes.uint8, [255, 254, 253, 252])

  def test_int8_to_uint16_negative(self):
    _test_op(lambda: Tensor([-1, -2, -3, -4], dtype=dtypes.int8).cast(dtypes.uint16), dtypes.uint16, [2**16-1, 2**16-2, 2**16-3, 2**16-4])

  @unittest.skipIf(getenv("PTX"), "broken in ptx")
  def test_bitcast_alt(self):
    a = Tensor([72, -90, 27, 40, -53, 70, 96, 51], dtype=dtypes.int8).bitcast(dtypes.short)
    self.assertListEqual(a.tolist(), [-22968, 10267, 18123, 13152])

class TestUint8DType(TestDType):
  DTYPE = dtypes.uint8
  @unittest.skipIf(getenv("CUDA",0)==1 or getenv("PTX", 0)==1, "cuda saturation works differently")
  def test_uint8_to_int8_overflow(self):
    _test_op(lambda: Tensor([255, 254, 253, 252], dtype=dtypes.uint8).cast(dtypes.int8), dtypes.int8, [-1, -2, -3, -4])

@unittest.skipIf(Device.DEFAULT == "WEBGL", "No bitcast on WebGL")
class TestBitCast(unittest.TestCase):
  @given(strat.sampled_from(dtype_ints + dtype_floats), strat.sampled_from(dtype_ints + dtype_floats))
  def test_shape_change_bitcast(self, dt1, dt2):
    # NOTE: this has to be assume to prevent hypothesis from skipping all samples
    assume(dt2 != dtypes.bfloat16 and dt1 != dtypes.bfloat16) # no test for bf16 bitcast yet
    assume(not (getenv("PTX") and dt1 == dtypes.int8)) # TODO: bitcasting int8 fails in PTX
    data = rand_for_dtype(dt1, 32).reshape(2, 2, 8)
    _test_op(lambda: Tensor(data, dtype=dt1).bitcast(dt2), dt2, data.view(_to_np_dtype(dt2)).tolist())

  def test_shape_change_bitcast_exceptions(self):
    with self.assertRaises(RuntimeError):
      # should fail because 3 int8 is 3 bytes but float16 is two and 3 isn't a multiple of 2
      Tensor.empty((3,), dtype=dtypes.int8).bitcast(dtypes.float16)

    with self.assertRaises(RuntimeError):
      # should fail because backprop through bitcast is undefined
      Tensor.empty((4,), dtype=dtypes.int8, requires_grad=True).bitcast(dtypes.float16)

  def test_bitcast_float_to_int32(self):
    a = Tensor([1.,2,3])
    b = a.bitcast(dtypes.int32)
    assert b.numpy()[0] == 0x3f800000

  def test_bitcast_upcasted(self):
    a = Tensor.zeros(100, 4, dtype=dtypes.int32).contiguous() + 0x3f800000
    b = a.bitcast(dtypes.float32)
    assert b.numpy()[0,0] == 1.

class TestInt16DType(TestDType): DTYPE = dtypes.int16

class TestUint16DType(TestDType):
  DTYPE = dtypes.uint16

  def test_uint16_to_int8_overflow(self):
    _test_op(lambda: Tensor([2**16-1, 2**16-2, 1, 0], dtype=dtypes.uint16).cast(dtypes.int8), dtypes.int8, [-1, -2, 1, 0])

class TestInt32DType(TestDType): DTYPE = dtypes.int32
class TestUint32DType(TestDType): DTYPE = dtypes.uint32

class TestInt64DType(TestDType): DTYPE = dtypes.int64
class TestUint64DType(TestDType):
  DTYPE = dtypes.uint64
  def test_uint64_load(self):
    assert Tensor(2**64 - 1, dtype=dtypes.uint64).numpy() == 2**64 - 1

class TestBoolDType(TestDType): DTYPE = dtypes.bool

class TestPtrDType(unittest.TestCase):
  def test_vec_double(self):
    dt1 = dtypes.float.vec(4).ptr().vec(4)
    dt2 = dtypes.float.vec(4).ptr().vec(4)
    self.assertEqual(dt1, dt2)
    self.assertEqual(str(dt1), str(dt2))

  def test_scalar(self):
    dt = dtypes.float.vec(4).ptr().scalar()
    self.assertEqual(dt.base, dtypes.float.vec(4))

    dt = dtypes.float.vec(4).ptr().vec(4).scalar()
    self.assertEqual(dt.base, dtypes.float.vec(4))

    dt = dtypes.float.vec(4).scalar()
    self.assertEqual(dt, dtypes.float)

  def test_serialize(self):
    dt = dtypes.float.vec(4).ptr().vec(4)
    self.assertEqual(dt, eval(str(dt)))

  def test_vec_ptr_sz(self):
    dt = dtypes.float.ptr(1024).vec(4)
    self.assertEqual(dt, eval(str(dt)))
    self.assertEqual(str(dt), "dtypes.float.ptr(1024).vec(4)")

  def test_vcount(self):
    dt = dtypes.float.ptr().vec(4)
    self.assertEqual(dt.vcount, 4)
    self.assertEqual(dt.v, 4)
    self.assertEqual(dt.count, 1)

    dt = dtypes.float.vec(4).ptr()
    self.assertEqual(dt.vcount, 1)
    self.assertEqual(dt.v, 1)
    self.assertEqual(dt.count, 4)

    dt = dtypes.float.vec(4).ptr().vec(4)
    self.assertEqual(dt.vcount, 4)
    self.assertEqual(dt.v, 4)
    self.assertEqual(dt.count, 4)

class TestImplicitFunctionTypeChange(unittest.TestCase):
  def test_functions(self):
    result = []
    for func in [
      lambda t: t.exp(),
      lambda t: t.exp2(),
      lambda t: t.log(),
      lambda t: t.log2(),
      lambda t: t.sqrt(),
      lambda t: t.sin(),
    ]:
      t = func(Tensor([4.0, 3.0])).max() == func(Tensor([4.0, 3.0]))
      result.append(t.numpy().sum())
    assert all(result)

class TestTensorMethod(unittest.TestCase):
  @given(strat.sampled_from(core_dtypes))
  def test_abs_diff(self, dt):
    if dt == dtypes.bool or not is_dtype_supported(dt): return
    a, b = Tensor([2], dtype=dt), Tensor([1], dtype=dt)
    ret = (a - b).abs()
    np.testing.assert_allclose(ret.numpy(), np.abs(a.numpy()-b.numpy()))

class TestDtypeUsage(unittest.TestCase):
  def test_max_w_alu(self):
    for d in dtypes.ints:
      if is_dtype_supported(d):
        t = Tensor([[1, 2], [3, 4]], dtype=d)
        (t*t).max().item()

@unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), f"no bfloat16 on {Device.DEFAULT}")
class TestOpsBFloat16(unittest.TestCase):
  def test_cast(self):
    # TODO: helper_test_op breaks in unrelated part
    # TODO: wrong output with GPU=1 / PYTHON=1 on mac
    data = [60000.0, 70000.0, 80000.0]
    np.testing.assert_allclose(Tensor(data).cast("bfloat16").numpy(), torch.tensor(data).type(torch.bfloat16).float().numpy())

if __name__ == '__main__':
  unittest.main()
