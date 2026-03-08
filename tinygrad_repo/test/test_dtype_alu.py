import unittest, operator, math
from tinygrad import Context, Tensor, dtypes, Device
from tinygrad.dtype import DType, truncate
from tinygrad.helpers import CI, getenv
from tinygrad.tensor import _to_np_dtype
from tinygrad.device import is_dtype_supported
from tinygrad.runtime.ops_python import from_storage_scalar
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.nir import NIRRenderer
from tinygrad.uop import Ops
import numpy as np
import pytest
from hypothesis import assume, given, strategies as strat, settings

pytestmark = pytest.mark.filterwarnings("ignore")

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")
print(settings.default)

dtypes_float = (dtypes.float16, dtypes.float32, dtypes.float64)
dtypes_int = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
dtypes_bool = (dtypes.bool,)
binary_operations = [operator.add, operator.sub, operator.mul, operator.lt, operator.eq]

integer_binary_operations = binary_operations + [(Tensor.bitwise_xor, np.bitwise_xor), (Tensor.bitwise_and, np.bitwise_and),
                                                 (Tensor.bitwise_or, np.bitwise_or), (Tensor.maximum, np.maximum), operator.mod]
integer_unary_operations = [operator.neg]
unary_operations = [(Tensor.exp, np.exp), (Tensor.log, np.log), (Tensor.sin, np.sin),
                    (Tensor.sqrt, np.sqrt), (Tensor.reciprocal, np.reciprocal), (Tensor.cos, np.cos)]

# TODO: enable this (this is a dtype issue)
#binary_operations.append(operator.truediv)

# TODO: CI CUDA segfaults on sin, WEBGPU and NIR sines are not precise enough for large numbers
if (getenv("MOCKGPU") and Device.DEFAULT in {"NV", "CUDA"}) or Device.DEFAULT == "WEBGPU" or isinstance(Device[Device.DEFAULT].renderer, NIRRenderer):
  unary_operations.remove((Tensor.sin, np.sin))
  unary_operations.remove((Tensor.cos, np.cos))

class ht:
  float64 = strat.floats(width=64, allow_subnormal=False)
  float32 = strat.floats(width=32, allow_subnormal=False)
  float16 = strat.floats(width=16, allow_subnormal=False)
  uint8 = strat.integers(0, 255)
  uint16 = strat.integers(0, 65535)
  uint32 = strat.integers(0, 2**32-1)
  uint64 = strat.integers(0, 2**64-1)
  int8 = strat.integers(-128, 127)
  int16 = strat.integers(-32768, 32767)
  int32 = strat.integers(-2147483648, 2147483647)
  int64 = strat.integers(-9223372036854775808, 9223372036854775807)
  bool = strat.booleans()
ht.bfloat16 = ht.uint16.filter(lambda x: ((x >> 7) & 0xFF) != 0)  # filter subnormal bfloat16
ht.fp8e4m3 = ht.uint8
ht.fp8e5m2 = ht.uint8

def universal_test(a, b, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  if op[0] == operator.mod and b == 0: return
  # lt and max with nan is undefined in tinygrad
  if op[0] in (operator.lt, Tensor.maximum) and (math.isnan(a) or math.isnan(b)): return
  ta, tb = Tensor([a], dtype=dtype), Tensor([b], dtype=dtype)
  tensor_value = (op[0](ta, tb)).numpy()
  numpy_value = op[1](ta.numpy(), tb.numpy())
  if dtype in dtypes.fp8s: numpy_value = truncate[dtype](numpy_value.item())
  if dtype in dtypes.floats:
    atol, rtol = {dtypes.bfloat16:(1e-3, 1e-2), dtypes.fp8e4m3:(1e-1, 1e-1), dtypes.fp8e5m2:(1.0, 5e-1)}.get(dtype, (1e-10, 1e-7))
    np.testing.assert_allclose(tensor_value, numpy_value, atol=atol, rtol=rtol)
  else: np.testing.assert_equal(tensor_value, numpy_value)

def universal_test_unary(a, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  ta = Tensor([a], dtype=dtype)
  # TODO: cos does not match for large input
  if op[0] == Tensor.cos and abs(a) > 30: return
  if op[0] == Tensor.log and a <= 0: return
  out: Tensor = op[0](ta)
  tensor_value = out.numpy()
  numpy_value = op[1](ta.numpy())
  if dtype in dtypes.fp8s:
    # cuda cast f32 inf to f8 MAX, amd cast it to nan(E4M3)/inf(E5M2)
    if math.isinf(numpy_value.item()): return
    numpy_value = truncate[dtype](numpy_value.item())
  if dtype in dtypes.floats:
    atol, rtol = { dtypes.float16:(1e-3, 1e-2), dtypes.bfloat16:(1e-3, 2e-2),
      dtypes.fp8e4m3:(1e-1, 1e-1), dtypes.fp8e5m2: (1.0, 5e-1)}.get(dtype, (1e-6, 1e-5))
    np.testing.assert_allclose(tensor_value, numpy_value, atol=atol, rtol=rtol)
  else: np.testing.assert_equal(tensor_value, numpy_value)

def universal_test_cast(a, in_dtype, dtype):
  tensor_value = Tensor([a], dtype=in_dtype).cast(dtype)
  numpy_value = np.array([a], dtype=_to_np_dtype(in_dtype)).astype(_to_np_dtype(dtype))
  np.testing.assert_equal(tensor_value.numpy(), numpy_value)

@unittest.skipIf(Device.DEFAULT == "WEBGPU", "Inf and nan cases are wrong on WebGPU")
def universal_test_midcast(a, b, c, op1, op2, d1:DType, d2:DType):
  if not isinstance(op1, tuple): op1 = (op1, op1)
  if not isinstance(op2, tuple): op2 = (op2, op2)
  # lt and max with nan is undefined in tinygrad
  if op1[0] in (operator.lt, Tensor.maximum) and (math.isnan(a) or math.isnan(b)): return
  if op2[0] in (operator.lt, Tensor.maximum) and math.isnan(c): return
  at, bt, ct = Tensor([a], dtype=d1), Tensor([b], dtype=d1), Tensor([c], dtype=d2)
  an, bn, cn = np.array([a]).astype(_to_np_dtype(d1)), np.array([b]).astype(_to_np_dtype(d1)), np.array([c]).astype(_to_np_dtype(d2))
  tensor_value = op2[0](op1[0](at, bt).cast(d2), ct).numpy()
  numpy_value = op2[1](op1[1](an, bn).astype(_to_np_dtype(d2)), cn)
  np.testing.assert_allclose(tensor_value, numpy_value, rtol=1e-6 if isinstance(Device[Device.DEFAULT].renderer, PTXRenderer) else 1e-7)

class TestDTypeALU(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.float64), f"no float64 on {Device.DEFAULT}")
  @given(ht.float64, ht.float64, strat.sampled_from(binary_operations))
  def test_float64(self, a, b, op): universal_test(a, b, dtypes.float64, op)

  @given(ht.float32, ht.float32, strat.sampled_from(binary_operations))
  def test_float32(self, a, b, op): universal_test(a, b, dtypes.float32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, ht.float16, strat.sampled_from(binary_operations))
  def test_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), f"no bfloat16 on {Device.DEFAULT}")
  @given(ht.bfloat16, ht.bfloat16, strat.sampled_from(binary_operations))
  def test_bfloat16(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.bfloat16), from_storage_scalar(a, dtypes.bfloat16), dtypes.bfloat16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), f"no fp8e4m3 on {Device.DEFAULT}")
  @given(ht.fp8e4m3, ht.fp8e4m3, strat.sampled_from(binary_operations))
  def test_fp8e4m3(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e4m3), from_storage_scalar(b, dtypes.fp8e4m3), dtypes.fp8e4m3, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e5m2), f"no fp8e5m2 on {Device.DEFAULT}")
  @given(ht.fp8e5m2, ht.fp8e5m2, strat.sampled_from(binary_operations))
  def test_fp8e5m2(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e5m2), from_storage_scalar(b, dtypes.fp8e5m2), dtypes.fp8e5m2, op)

  @given(ht.float32, strat.sampled_from(unary_operations))
  def test_float32_unary(self, a, op): universal_test_unary(a, dtypes.float32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, strat.sampled_from(unary_operations))
  def test_float16_unary(self, a, op): universal_test_unary(a, dtypes.float16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), f"no bfloat16 on {Device.DEFAULT}")
  @given(ht.bfloat16, strat.sampled_from(unary_operations))
  def test_bfloat16_unary(self, a, op): universal_test_unary(from_storage_scalar(a, dtypes.bfloat16), dtypes.bfloat16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e4m3), f"no fp8e4m3 on {Device.DEFAULT}")
  @given(ht.fp8e4m3, strat.sampled_from(unary_operations))
  def test_fp8e4m3_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e4m3) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e4m3), dtypes.fp8e4m3, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.fp8e5m2), f"no fp8e5m2 on {Device.DEFAULT}")
  @given(ht.fp8e5m2, strat.sampled_from(unary_operations))
  def test_fp8e5m2_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e5m2) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e5m2), dtypes.fp8e5m2, op)

  @given(ht.uint8, ht.uint8, strat.sampled_from(integer_binary_operations))
  def test_uint8(self, a, b, op): universal_test(a, b, dtypes.uint8, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint16), f"no uint16 on {Device.DEFAULT}")
  @given(ht.uint16, ht.uint16, strat.sampled_from(integer_binary_operations))
  def test_uint16(self, a, b, op): universal_test(a, b, dtypes.uint16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint32), f"no uint32 on {Device.DEFAULT}")
  @given(ht.uint32, ht.uint32, strat.sampled_from(integer_binary_operations))
  def test_uint32(self, a, b, op): universal_test(a, b, dtypes.uint32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint64), f"no uint64 on {Device.DEFAULT}")
  @given(ht.uint64, ht.uint64, strat.sampled_from(integer_binary_operations))
  def test_uint64(self, a, b, op): universal_test(a, b, dtypes.uint64, op)

  @unittest.skipUnless(Ops.SHL in Device[Device.DEFAULT].renderer.code_for_op, "long decomp requires bitshift")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does indexing math with longs")
  @given(ht.uint64, ht.uint64, strat.sampled_from(integer_binary_operations))
  @Context(EMULATED_DTYPES="long")
  def test_emulated_uint64(self, a, b, op): universal_test(a, b, dtypes.uint64, op)

  @given(ht.int8, ht.int8, strat.sampled_from(integer_binary_operations))
  def test_int8(self, a, b, op): universal_test(a, b, dtypes.int8, op)

  @given(ht.int16, ht.int16, strat.sampled_from(integer_binary_operations))
  def test_int16(self, a, b, op): universal_test(a, b, dtypes.int16, op)

  @given(ht.int32, ht.int32, strat.sampled_from(integer_binary_operations))
  def test_int32(self, a, b, op): universal_test(a, b, dtypes.int32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.int64), f"no int64 on {Device.DEFAULT}")
  @given(ht.int64, ht.int64, strat.sampled_from(integer_binary_operations))
  def test_int64(self, a, b, op): universal_test(a, b, dtypes.int64, op)

  @unittest.skipUnless(Ops.SHL in Device[Device.DEFAULT].renderer.code_for_op, "long decomp requires bitshift")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does indexing math with longs")
  @given(ht.int64, ht.int64, strat.sampled_from(integer_binary_operations))
  @Context(EMULATED_DTYPES="long")
  def test_emulated_int64(self, a, b, op): universal_test(a, b, dtypes.int64, op)

  @given(ht.uint8, strat.sampled_from(integer_unary_operations))
  def test_uint8_unary(self, a, op): universal_test_unary(a, dtypes.uint8, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint16), f"no uint16 on {Device.DEFAULT}")
  @given(ht.uint16, strat.sampled_from(integer_unary_operations))
  def test_uint16_unary(self, a, op): universal_test_unary(a, dtypes.uint16, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint32), f"no uint32 on {Device.DEFAULT}")
  @given(ht.uint32, strat.sampled_from(integer_unary_operations))
  def test_uint32_unary(self, a, op): universal_test_unary(a, dtypes.uint32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.uint64), f"no uint64 on {Device.DEFAULT}")
  @given(ht.uint64, strat.sampled_from(integer_unary_operations))
  def test_uint64_unary(self, a, op): universal_test_unary(a, dtypes.uint64, op)

  @unittest.skipUnless(Ops.SHL in Device[Device.DEFAULT].renderer.code_for_op, "long decomp requires bitshift")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does indexing math with longs")
  @given(ht.uint64, strat.sampled_from(integer_unary_operations))
  @Context(EMULATED_DTYPES="long")
  def test_emulated_uint64_unary(self, a, op): universal_test_unary(a, dtypes.uint64, op)

  @given(ht.int8, strat.sampled_from(integer_unary_operations))
  def test_int8_unary(self, a, op): universal_test_unary(a, dtypes.int8, op)

  @given(ht.int16, strat.sampled_from(integer_unary_operations))
  def test_int16_unary(self, a, op): universal_test_unary(a, dtypes.int16, op)

  @given(ht.int32, strat.sampled_from(integer_unary_operations))
  def test_int32_unary(self, a, op): universal_test_unary(a, dtypes.int32, op)

  @unittest.skipUnless(is_dtype_supported(dtypes.int64), f"no int64 on {Device.DEFAULT}")
  @given(ht.int64, strat.sampled_from(integer_unary_operations))
  def test_int64_unary(self, a, op): universal_test_unary(a, dtypes.int64, op)

  @unittest.skipUnless(Ops.SHL in Device[Device.DEFAULT].renderer.code_for_op, "long decomp requires bitshift")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does indexing math with longs")
  @given(ht.int64, strat.sampled_from(integer_unary_operations))
  @Context(EMULATED_DTYPES="long")
  def test_emulated_int64_unary(self, a, op): universal_test_unary(a, dtypes.int64, op)

  @given(ht.bool, ht.bool, strat.sampled_from(((operator.add, operator.add), (operator.mul, operator.mul))))
  def test_bool(self, a, b, op): universal_test(a, b, dtypes.bool, op)

  @unittest.skipIf(not CI and Device.DEFAULT == "METAL", "broken on local M3")
  @given(ht.int32, ht.int32, ht.float32, strat.sampled_from(integer_binary_operations), strat.sampled_from(binary_operations))
  def test_int32_midcast_float(self, a, b, c, op1, op2): universal_test_midcast(a, b, c, op1, op2, dtypes.int32, dtypes.float32)

  # Metal and CUDA and HIP and NIR behave differently than numpy in CI for overflows
  skip_overflow = (CI and Device.DEFAULT in {"AMD", "NV", "CUDA"}) or isinstance(Device[Device.DEFAULT].renderer, NIRRenderer)
  @given(strat.floats(width=32, min_value=0, max_value=10.0) if skip_overflow else ht.float32,
         strat.floats(width=32, min_value=0, max_value=10.0) if skip_overflow else ht.float32,
         ht.int32, strat.sampled_from(binary_operations), strat.sampled_from(integer_binary_operations))
  @unittest.skipIf(Device.DEFAULT == "PYTHON", "TODO: fix cast inf to int32 in PYTHON")
  @unittest.skip("broken on Mac")
  def test_float_midcast_int32(self, a, b, c, op1, op2): universal_test_midcast(a, b, c, op1, op2, dtypes.float32, dtypes.int32)

  @unittest.skip("broken. TODO: fix it")
  @given(ht.float32, strat.sampled_from(dtypes_float+dtypes_int+dtypes_bool))
  def test_float_cast(self, a, dtype): universal_test_cast(a, dtypes.float32, dtype)

  @unittest.skip("broken. TODO: fix it")
  @given(ht.int32, strat.sampled_from(dtypes_float+dtypes_int+dtypes_bool))
  def test_int32_cast(self, a, dtype): universal_test_cast(a, dtypes.int32, dtype)

  @given(strat.floats(width=32, min_value=1.0, max_value=254.0, allow_subnormal=False),
         strat.sampled_from(dtypes_float), strat.sampled_from((dtypes.uint8, dtypes.uint16)))
  def test_float_cast_to_unsigned(self, a, float_dtype, unsigned_dtype):
    if not is_dtype_supported(float_dtype): float_dtype = dtypes.float32
    universal_test_cast(a, float_dtype, unsigned_dtype)

  @given(strat.floats(width=32, min_value=256.0, max_value=65000.0, allow_subnormal=False),
         strat.sampled_from(dtypes_float), strat.sampled_from((dtypes.uint8, dtypes.uint16)))
  def test_float_cast_to_unsigned_overflow(self, a, float_dtype, unsigned_dtype):
    if not is_dtype_supported(float_dtype): float_dtype = dtypes.float32
    universal_test_cast(a, float_dtype, unsigned_dtype)

  @given(strat.floats(width=32, min_value=-65000.0, max_value=-1.0, allow_subnormal=False),
         strat.sampled_from(dtypes_float), strat.sampled_from((dtypes.uint8, dtypes.uint16)))
  def test_float_cast_to_unsigned_underflow(self, a, float_dtype, unsigned_dtype):
    if not is_dtype_supported(float_dtype): float_dtype = dtypes.float32
    universal_test_cast(a, float_dtype, unsigned_dtype)

  @unittest.expectedFailure
  def test_unsafe_cast_float_to_int_failure(self):
    val = float(dtypes.max(dtypes.int32) - 1)
    t1 = Tensor([val], dtype=dtypes.float32).cast(dtypes.int32)
    t2 = Tensor(val, dtype=dtypes.float32).cast(dtypes.int32)
    np.testing.assert_equal(t1.item(), t2.item())

if __name__ == '__main__':
  unittest.main()
