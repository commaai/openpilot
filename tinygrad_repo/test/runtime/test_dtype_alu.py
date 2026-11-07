import unittest, operator, math
from tinygrad import Context, Tensor, dtypes, Device
from tinygrad.dtype import DType, truncate, fp8_to_float
from tinygrad.helpers import EMULATED_DTYPES, DEV, getenv
from tinygrad.tensor import _to_np_dtype
from tinygrad.runtime.ops_python import from_storage_scalar
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.renderer.nir import NIRRenderer
from tinygrad.renderer.llvmir import CPULLVMRenderer
from tinygrad.renderer.x86 import X86Renderer
from tinygrad.uop import Ops
import numpy as np
import pytest
from hypothesis import assume, given, strategies as strat, settings

pytestmark = pytest.mark.filterwarnings("ignore")

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

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
if ((DEV.interface.startswith("MOCK") and Device.DEFAULT in {"NV", "CUDA"})
    or Device.DEFAULT == "WEBGPU" or isinstance(Device[Device.DEFAULT].renderer, NIRRenderer)):
  unary_operations.remove((Tensor.sin, np.sin))
  unary_operations.remove((Tensor.cos, np.cos))

# transcendental isn't accurate enough
if Ops.SQRT not in Device[Device.DEFAULT].renderer.code_for_op: unary_operations.remove((Tensor.sqrt, np.sqrt))

supported_dtypes = Device[Device.DEFAULT].renderer.supported_dtypes()

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
ht.fp8e4m3fnuz = ht.uint8
ht.fp8e5m2fnuz = ht.uint8

def universal_test(a, b, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  if op[0] == operator.mod and b == 0: return
  # TODO: throws floating point exception
  if isinstance(Device[Device.DEFAULT].renderer, (X86Renderer, CPULLVMRenderer)) and op[0] == operator.mod and a == dtype.min and b == -1: return
  # lt and max with nan is undefined in tinygrad
  if op[0] in (operator.lt, Tensor.maximum) and (math.isnan(a) or math.isnan(b)): return
  ta, tb = Tensor([a], dtype=dtype), Tensor([b], dtype=dtype)
  if dtype in (*dtypes.fp8s, dtypes.bfloat16) and op[0] not in (operator.lt, operator.eq):
    storage_dtype = dtypes.uint16 if dtype == dtypes.bfloat16 else dtypes.uint8
    tensor_value = from_storage_scalar((op[0](ta.realize(), tb.realize())).bitcast(storage_dtype).item(), dtype)
    # numpy represents these dtypes as float32, so round the reference result too (including overflow).
    numpy_value = truncate[dtype](op[1](ta.numpy(), tb.numpy()).item())
  else: tensor_value, numpy_value = (op[0](ta, tb)).numpy(), op[1](ta.numpy(), tb.numpy())
  if dtype in dtypes.floats:
    if dtype not in supported_dtypes or dtype in EMULATED_DTYPES.tolist(dtypes): # denormals are zero
      fe, fm = dtypes.finfo(dtype)
      atol, rtol = 2 ** (2 - (1 << (fe - 1))), 2 ** (-fm)
    else: atol, rtol = {dtypes.bfloat16:(1e-3, 1e-2), dtypes.fp8e4m3:(1e-1, 1e-1), dtypes.fp8e5m2:(1.0, 5e-1),
                        dtypes.fp8e4m3fnuz:(1e-1, 1e-1), dtypes.fp8e5m2fnuz:(5e-1, 5e-1)}.get(dtype, (1e-10, 1e-7))
    np.testing.assert_allclose(tensor_value, numpy_value, atol=atol, rtol=rtol)
  else: np.testing.assert_equal(tensor_value, numpy_value)

def universal_test_unary(a, dtype, op):
  if not isinstance(op, tuple): op = (op, op)
  ta = Tensor([a], dtype=dtype)
  # TODO: cos does not match for large input
  if op[0] == Tensor.cos and abs(a) > 30: return
  if op[0] == Tensor.log and a <= 0: return
  if dtype in dtypes.fp8s:
    # denormals are zero
    if (dtype in EMULATED_DTYPES.tolist(dtypes) or dtype not in supported_dtypes
        and abs(ta.numpy().item()) < 0.015625): return
    tensor_value = fp8_to_float(op[0](ta.realize()).bitcast(dtypes.uint8).item(), dtype)
    numpy_value = truncate[dtype](v:=op[1](ta.numpy()).item())
    # cuda cast f32 inf to f8 MAX, amd cast it to nan(E4M3)/inf(E5M2)
    if math.isinf(v): return
  else: tensor_value, numpy_value = op[0](ta).numpy(), op[1](ta.numpy())
  if dtype in dtypes.floats:
    atol, rtol = { dtypes.float16:(1e-3, 1e-2), dtypes.bfloat16:(1e-3, 2e-2),
      dtypes.fp8e4m3:(1e-1, 1e-1), dtypes.fp8e5m2: (1.0, 5e-1),
      dtypes.fp8e4m3fnuz:(1e-1, 1e-1), dtypes.fp8e5m2fnuz: (5e-1, 5e-1)}.get(dtype, (1e-6, 1e-5))
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
  if op1[0] == operator.mod and b == 0: return
  # lt and max with nan is undefined in tinygrad
  if op1[0] in (operator.lt, Tensor.maximum) and (math.isnan(a) or math.isnan(b)): return
  if op2[0] in (operator.lt, Tensor.maximum) and math.isnan(c): return
  at, bt, ct = Tensor([a], dtype=d1), Tensor([b], dtype=d1), Tensor([c], dtype=d2)
  an, bn, cn = np.array([a]).astype(_to_np_dtype(d1)), np.array([b]).astype(_to_np_dtype(d1)), np.array([c]).astype(_to_np_dtype(d2))
  tensor_value = op2[0](op1[0](at, bt).cast(d2), ct).numpy()
  numpy_value = op2[1](op1[1](an, bn).astype(_to_np_dtype(d2)), cn)
  np.testing.assert_allclose(tensor_value, numpy_value, rtol=1e-6 if isinstance(Device[Device.DEFAULT].renderer, PTXRenderer) else 1e-7)

class TestDTypeALU(unittest.TestCase):
  @unittest.skipUnless(dtypes.float64 in supported_dtypes, f"no float64 on {Device.DEFAULT}")
  @given(ht.float64, ht.float64, strat.sampled_from(binary_operations))
  def test_float64(self, a, b, op): universal_test(a, b, dtypes.float64, op)

  @given(ht.float32, ht.float32, strat.sampled_from(binary_operations))
  def test_float32(self, a, b, op): universal_test(a, b, dtypes.float32, op)

  @unittest.skipUnless(dtypes.float16 in supported_dtypes, f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, ht.float16, strat.sampled_from(binary_operations))
  def test_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @given(ht.float16, ht.float16, strat.sampled_from(binary_operations))
  @Context(EMULATED_DTYPES="half")
  def test_emulated_float16(self, a, b, op): universal_test(a, b, dtypes.float16, op)

  @unittest.skipUnless(dtypes.bfloat16 in supported_dtypes, f"no bfloat16 on {Device.DEFAULT}")
  @given(ht.bfloat16, ht.bfloat16, strat.sampled_from(binary_operations))
  def test_bfloat16(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.bfloat16), from_storage_scalar(b, dtypes.bfloat16), dtypes.bfloat16, op)

  @given(ht.bfloat16, ht.bfloat16, strat.sampled_from(binary_operations))
  @Context(EMULATED_DTYPES="bfloat16")
  def test_emulated_bfloat16(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.bfloat16), from_storage_scalar(b, dtypes.bfloat16), dtypes.bfloat16, op)

  @unittest.skipUnless(dtypes.fp8e4m3 in supported_dtypes, f"no fp8e4m3 on {Device.DEFAULT}")
  @given(ht.fp8e4m3, ht.fp8e4m3, strat.sampled_from(binary_operations))
  def test_fp8e4m3(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e4m3), from_storage_scalar(b, dtypes.fp8e4m3), dtypes.fp8e4m3, op)

  @given(ht.fp8e4m3, ht.fp8e4m3, strat.sampled_from(binary_operations))
  @Context(EMULATED_DTYPES="fp8e4m3")
  def test_emulated_fp8e4m3(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e4m3), from_storage_scalar(b, dtypes.fp8e4m3), dtypes.fp8e4m3, op)

  @unittest.skipUnless(dtypes.fp8e5m2 in supported_dtypes, f"no fp8e5m2 on {Device.DEFAULT}")
  @given(ht.fp8e5m2, ht.fp8e5m2, strat.sampled_from(binary_operations))
  def test_fp8e5m2(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e5m2), from_storage_scalar(b, dtypes.fp8e5m2), dtypes.fp8e5m2, op)

  @given(ht.fp8e5m2, ht.fp8e5m2, strat.sampled_from(binary_operations))
  @Context(EMULATED_DTYPES="fp8e5m2")
  def test_emulated_fp8e5m2(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e5m2), from_storage_scalar(b, dtypes.fp8e5m2), dtypes.fp8e5m2, op)

  @unittest.skipUnless(dtypes.fp8e4m3fnuz in supported_dtypes, f"no fp8e4m3fnuz on {Device.DEFAULT}")
  @given(ht.fp8e4m3fnuz, ht.fp8e4m3fnuz, strat.sampled_from(binary_operations))
  def test_fp8e4m3fnuz(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e4m3fnuz), from_storage_scalar(b, dtypes.fp8e4m3fnuz), dtypes.fp8e4m3fnuz, op)

  @unittest.skipUnless(dtypes.fp8e5m2fnuz in supported_dtypes, f"no fp8e5m2fnuz on {Device.DEFAULT}")
  @given(ht.fp8e5m2fnuz, ht.fp8e5m2fnuz, strat.sampled_from(binary_operations))
  def test_fp8e5m2fnuz(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e5m2fnuz), from_storage_scalar(b, dtypes.fp8e5m2fnuz), dtypes.fp8e5m2fnuz, op)

  @given(ht.fp8e4m3fnuz, ht.fp8e4m3fnuz, strat.sampled_from(binary_operations))
  @Context(EMULATED_DTYPES="fp8e4m3fnuz")
  def test_emulated_fp8e4m3fnuz(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e4m3fnuz), from_storage_scalar(b, dtypes.fp8e4m3fnuz), dtypes.fp8e4m3fnuz, op)

  @given(ht.fp8e5m2fnuz, ht.fp8e5m2fnuz, strat.sampled_from(binary_operations))
  @Context(EMULATED_DTYPES="fp8e5m2fnuz")
  def test_emulated_fp8e5m2fnuz(self, a, b, op):
    universal_test(from_storage_scalar(a, dtypes.fp8e5m2fnuz), from_storage_scalar(b, dtypes.fp8e5m2fnuz), dtypes.fp8e5m2fnuz, op)

  @given(ht.float32, strat.sampled_from(unary_operations))
  def test_float32_unary(self, a, op): universal_test_unary(a, dtypes.float32, op)

  @unittest.skipUnless(dtypes.float16 in supported_dtypes, f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, strat.sampled_from(unary_operations))
  def test_float16_unary(self, a, op): universal_test_unary(a, dtypes.float16, op)

  @given(ht.float16, strat.sampled_from(unary_operations))
  @Context(EMULATED_DTYPES="half")
  def test_emulated_float16_unary(self, a, op): universal_test_unary(a, dtypes.float16, op)

  @unittest.skipUnless(dtypes.bfloat16 in supported_dtypes, f"no bfloat16 on {Device.DEFAULT}")
  @given(ht.bfloat16, strat.sampled_from(unary_operations))
  def test_bfloat16_unary(self, a, op): universal_test_unary(from_storage_scalar(a, dtypes.bfloat16), dtypes.bfloat16, op)

  @given(ht.bfloat16, strat.sampled_from(unary_operations))
  @Context(EMULATED_DTYPES="bfloat16")
  def test_emulated_bfloat16_unary(self, a, op): universal_test_unary(from_storage_scalar(a, dtypes.bfloat16), dtypes.bfloat16, op)

  @unittest.skipUnless(dtypes.fp8e4m3 in supported_dtypes, f"no fp8e4m3 on {Device.DEFAULT}")
  @given(ht.fp8e4m3, strat.sampled_from(unary_operations))
  def test_fp8e4m3_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e4m3) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e4m3), dtypes.fp8e4m3, op)

  @given(ht.fp8e4m3, strat.sampled_from(unary_operations))
  @Context(EMULATED_DTYPES="fp8e4m3")
  def test_emulated_fp8e4m3_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e4m3) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e4m3), dtypes.fp8e4m3, op)

  @unittest.skipUnless(dtypes.fp8e5m2 in supported_dtypes, f"no fp8e5m2 on {Device.DEFAULT}")
  @given(ht.fp8e5m2, strat.sampled_from(unary_operations))
  def test_fp8e5m2_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e5m2) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e5m2), dtypes.fp8e5m2, op)

  @given(ht.fp8e5m2, strat.sampled_from(unary_operations))
  @Context(EMULATED_DTYPES="fp8e5m2")
  def test_emulated_fp8e5m2_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e5m2) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e5m2), dtypes.fp8e5m2, op)

  @unittest.skipUnless(dtypes.fp8e4m3fnuz in supported_dtypes, f"no fp8e4m3fnuz on {Device.DEFAULT}")
  @given(ht.fp8e4m3fnuz, strat.sampled_from(unary_operations))
  def test_fp8e4m3fnuz_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e4m3fnuz) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e4m3fnuz), dtypes.fp8e4m3fnuz, op)

  @unittest.skipUnless(dtypes.fp8e5m2fnuz in supported_dtypes, f"no fp8e5m2fnuz on {Device.DEFAULT}")
  @given(ht.fp8e5m2fnuz, strat.sampled_from(unary_operations))
  def test_fp8e5m2fnuz_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e5m2fnuz) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e5m2fnuz), dtypes.fp8e5m2fnuz, op)

  @given(ht.fp8e4m3fnuz, strat.sampled_from(unary_operations))
  @Context(EMULATED_DTYPES="fp8e4m3fnuz")
  def test_emulated_fp8e4m3fnuz_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e4m3fnuz) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e4m3fnuz), dtypes.fp8e4m3fnuz, op)

  @given(ht.fp8e5m2fnuz, strat.sampled_from(unary_operations))
  @Context(EMULATED_DTYPES="fp8e5m2fnuz")
  def test_emulated_fp8e5m2fnuz_unary(self, a, op):
    if op[1] == np.reciprocal: assume(from_storage_scalar(a, dtype=dtypes.fp8e5m2fnuz) != 0.0)
    universal_test_unary(from_storage_scalar(a, dtype=dtypes.fp8e5m2fnuz), dtypes.fp8e5m2fnuz, op)

  @given(ht.uint8, ht.uint8, strat.sampled_from(integer_binary_operations))
  def test_uint8(self, a, b, op): universal_test(a, b, dtypes.uint8, op)

  @unittest.skipUnless(dtypes.uint16 in supported_dtypes, f"no uint16 on {Device.DEFAULT}")
  @given(ht.uint16, ht.uint16, strat.sampled_from(integer_binary_operations))
  def test_uint16(self, a, b, op): universal_test(a, b, dtypes.uint16, op)

  @unittest.skipUnless(dtypes.uint32 in supported_dtypes, f"no uint32 on {Device.DEFAULT}")
  @given(ht.uint32, ht.uint32, strat.sampled_from(integer_binary_operations))
  def test_uint32(self, a, b, op): universal_test(a, b, dtypes.uint32, op)

  @unittest.skipUnless(dtypes.uint64 in supported_dtypes, f"no uint64 on {Device.DEFAULT}")
  @given(ht.uint64, ht.uint64, strat.sampled_from(integer_binary_operations))
  def test_uint64(self, a, b, op): universal_test(a, b, dtypes.uint64, op)

  def test_unsigned_add_lt(self):
    for dt in (dtypes.uint32, dtypes.uint64):
      x = Tensor.arange(10, dtype=dt) + Tensor([dt.max-4], dtype=dt)
      self.assertEqual((x < 3).tolist(), [i in (5, 6, 7) for i in range(10)])

  def test_unsigned_narrow_wrap(self):
    for dt in (dtypes.uint8, dtypes.uint16):
      x = Tensor([dt.max, 3], dtype=dt)
      self.assertEqual(((x+1)>>4).tolist(), [0, 0])
      self.assertEqual(((x-4)>>4).tolist(), [dt.max>>4, dt.max>>4])
      self.assertEqual(((x*2)>>4).tolist(), [dt.max>>4, 0])
      self.assertEqual(((-x)>>4).tolist(), [0, dt.max>>4])
      self.assertEqual(((x<<4)>>4).tolist(), [dt.max>>4, 3])
      self.assertEqual(((x*x)>>4).tolist(), [0, 0])
      self.assertEqual(((x+1)<1).tolist(), [True, False])

  def test_pow_one_keeps_dtype(self):
    self.assertEqual(((1**Tensor([2], dtype=dtypes.uint8))+255).cast(dtypes.uint32).tolist(), [0])
    self.assertEqual((1**Tensor([2], dtype=dtypes.int8)).bitcast(dtypes.uint8).tolist(), [1])

  def test_div_add_div_no_overflow(self):
    for dt in (dtypes.uint32, dtypes.uint64):
      x = Tensor([dt.max-1], dtype=dt)
      self.assertEqual(((x//2+1)//2).tolist(), [(dt.max+1)//4])
    x = Tensor([2**31-2], dtype=dtypes.int32)
    self.assertEqual(((x//2+2**29)//2).tolist(), [805306367])

  def test_div_div_no_overflow(self):
    self.assertEqual(((Tensor([2**31-1, -5], dtype=dtypes.int32)//65536)//65536).tolist(), [0, -1])

  def test_add_add_no_overflow(self):
    self.assertEqual((((Tensor([-2**31, 1-2**31], dtype=dtypes.int32)+2**30)+2**30)//2).tolist(), [0, 0])

  def test_neg_zero(self):
    self.assertEqual((1/-Tensor([0., -0.])).tolist(), [-math.inf, math.inf])

  def test_neg_add_half_zero(self):
    self.assertEqual((1/-(Tensor([0.])+Tensor([-0.])).half()).tolist(), [-math.inf])

  def test_neg_add_no_overflow(self):
    self.assertEqual((-(Tensor([1, 2], dtype=dtypes.int32)+(-2**31))==2**31-1).tolist(), [True, False])

  def test_neg_add_signed_zero(self):
    self.assertEqual((1/-(Tensor([0.])+Tensor([-0.]))).tolist(), [-math.inf])

  def test_add_mul_no_overflow(self):
    self.assertEqual((((Tensor([-715827887, 0], dtype=dtypes.int32)+5)*3)==-2**31+2).tolist(), [True, False])

  def test_add_lt_no_overflow(self):
    self.assertEqual(((Tensor([0, -715827882], dtype=dtypes.int32)*3+15) < -2**31+8).tolist(), [False, False])

  def test_mul_mul_no_overflow(self):
    self.assertEqual(((Tensor([-1, 0], dtype=dtypes.int32)*65536)*32768//2).tolist(), [-2**30, 0])

  def test_combine_terms_no_overflow(self):
    t, u = Tensor([-1, 0], dtype=dtypes.int32), Tensor([0, 0], dtype=dtypes.int32)
    self.assertEqual(((t*2**30+t*2**30)//2).tolist(), [-2**30, 0])
    self.assertEqual(((t+t*(2**31-1))//2).tolist(), [-2**30, 0])
    self.assertEqual((((u+t*(2**31-1))+t)//2).tolist(), [-2**30, 0])

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, CUDARenderer), "nvrtc 11.5 reassociates (x+c)+y<d and folds it to True")
  def test_reassociate_no_overflow(self):
    t, c = Tensor.arange(3, dtype=dtypes.int32), Tensor([False, True])
    a, b, x, y = (Tensor(v, dtype=dtypes.int32) for v in ([1, 0], [1, 1], [2, 0], [2**30, 1]))
    self.assertEqual((((t*2**29-2**30)+t*2**29) < 2**29).tolist(), [True, True, False])
    self.assertEqual((((t-2**30)+t*(2**30-1)) < 2**29).tolist(), [True, True, False])
    self.assertEqual((((a*2**30-2**30)+b*2**30) < 2**29).tolist(), [False, True])
    self.assertEqual((((x*-1)*y) < -1).tolist(), [True, False])
    self.assertEqual((((c.where(0, b*2**30)-2**30)+c.where(0, b*2**30)) < 2**29).tolist(), [False, True])

  def test_float_reassociate(self):
    x, y, c = Tensor([1e8]), Tensor([3.]), Tensor([True])
    self.assertEqual(((x-1e8)+Tensor([1.])).tolist(), [1.])
    self.assertEqual(((Tensor([1e30])*1e10)*Tensor([1e-30])).tolist(), [math.inf])
    self.assertEqual(((x+y)+y).tolist(), [1e8])
    self.assertEqual(((Tensor([1.])+(z:=Tensor([1e38]))*4)+z*-3).tolist(), [math.inf])
    self.assertEqual(((Tensor([1.])+z*-3.5)+z).tolist(), [-math.inf])
    self.assertEqual(((Tensor([1.])+c.where(1e8, x))+c.where(-1e8, x)).tolist(), [0.])

  @unittest.skipIf(Device.DEFAULT == "WEBGPU" or isinstance(Device[Device.DEFAULT].renderer, NIRRenderer), "backend compiler reassociates floats")
  def test_float_reassociate_consts(self): self.assertEqual(((Tensor([1.])+1e8)-1e8).tolist(), [0.])

  def test_fused_div_matches_unfused(self):
    x, y, z = Tensor([3e38, 1e30, 1.7]), Tensor([1e20, 1e30, 1.3]), Tensor([1e20, 1e10, 3.1])
    for f, g in ((lambda: x*y, lambda p: p/y), (lambda: x/y, lambda p: p/z), (lambda: z*z, lambda p: 1/p), (lambda: z*z*z, lambda p: 1/p)):
      np.testing.assert_equal(g(f()).numpy(), g(f().realize()).numpy())

  @unittest.skipIf(Device.DEFAULT == "CL", "rusticl fdiv scaling reassociates (x*c)*0.25 past inf")
  def test_fused_recip_const_matches_unfused(self):
    x = Tensor([3e38, 1e30, 1.7])
    np.testing.assert_equal((1/(x*3)).numpy(), (1/(x*3).realize()).numpy())

  def test_unsigned_mul_max(self):
    for dt in (dtypes.uint32, dtypes.uint64):
      x = Tensor([dt.max//2, dt.max//2+1], dtype=dt)
      self.assertEqual((x*2).max().item(), dt.max-1)

  def test_unsigned_wrap_bounds(self):
    for dt in (dtypes.uint32, dtypes.uint64):
      h = 1 << (dt.bitsize-1)
      u, x = Tensor([h], dtype=dt).maximum(h), Tensor([1], dtype=dt)
      self.assertEqual([(t < 1).item() for t in (u+h, u*2, u<<1, -x, x^-1)], [True, True, True, False, False])

  def test_unsigned_const_canonical(self):
    for dt in (dtypes.uint8, dtypes.uint32, dtypes.uint64):
      c, z = Tensor(-1, dtype=dt), Tensor([1], dtype=dt)*0-1
      self.assertEqual((c>>1).item(), dt.max>>1)
      self.assertEqual((z>>1).item(), dt.max>>1)
      self.assertEqual((z//2).item(), dt.max//2)
      self.assertEqual((z<3).item(), False)
      self.assertEqual(z.maximum(5).item(), dt.max)
    self.assertEqual(Tensor(-1, dtype=dtypes.uint32).cast(dtypes.int64).item(), 2**32-1)
    self.assertEqual((Tensor(255, dtype=dtypes.uint8)+1).cast(dtypes.uint16).item(), 0)
    self.assertEqual((Tensor([5], dtype=dtypes.uint32)//Tensor(-1, dtype=dtypes.uint32)).item(), 0)

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

  @given(ht.int64, ht.int64, strat.sampled_from(integer_binary_operations))
  def test_int64(self, a, b, op): universal_test(a, b, dtypes.int64, op)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does indexing math with longs")
  @given(ht.int64, ht.int64, strat.sampled_from(integer_binary_operations))
  @Context(EMULATED_DTYPES="long")
  def test_emulated_int64(self, a, b, op): universal_test(a, b, dtypes.int64, op)

  def _test_shl(self):
    for dtype, values, distances in ((dtypes.int64, [-0x1234, 0x80000001, -1, 0x1234, 1], [0, 5, 31, 32, 62]),
                                     (dtypes.uint64, [0x80000001, 0x80000001, 1, 0xFEDC, 1], [0, 5, 31, 32, 62]),
                                     (dtypes.int8, [-3, 1, 7, -2, 1], [0, 1, 3, 5, 6]),
                                     (dtypes.uint16, [3, 1, 0xFF, 7, 1], [0, 1, 7, 12, 15])):
      with self.subTest(dtype=dtype):
        result = Tensor(values, dtype=dtype) << Tensor(distances, dtype=dtype)
        np.testing.assert_equal(result.numpy(), [x << d for x, d in zip(values, distances)])

  def _test_shr(self):
    for dtype, values, distances in ((dtypes.int64, [-(2**40), -1, -(2**50), -(2**40), 0x123456789ABCDEF], [0, 5, 31, 32, 63]),
                                     (dtypes.uint64, [0xFEDCBA9876543210] * 5, [0, 5, 31, 32, 63]),
                                     (dtypes.int8, [-128, -1, 64, -37, 1], [0, 1, 3, 5, 7]),
                                     (dtypes.uint16, [0xFFFF] * 5, [0, 1, 8, 13, 15])):
      with self.subTest(dtype=dtype):
        result = Tensor(values, dtype=dtype) >> Tensor(distances, dtype=dtype)
        np.testing.assert_equal(result.numpy(), [x >> d for x, d in zip(values, distances)])

  def test_shl(self): self._test_shl()
  def test_shr(self): self._test_shr()
  def test_shr_long_of_small_value(self):
    self.assertEqual((Tensor([-1, 5], dtype=dtypes.int8).cast(dtypes.int64) >> 32).tolist(), [-1, 0])
  def test_mod_long_of_int_min(self):
    a, b = Tensor([-2**31, 5], dtype=dtypes.int32).cast(dtypes.int64), Tensor([-1, -2], dtype=dtypes.int8).minimum(-1).cast(dtypes.int64)
    self.assertEqual((a % b).tolist(), [0, -1])

  @Context(EMULATED_DTYPES="long")
  def test_emulated_shl(self): self._test_shl()

  @Context(EMULATED_DTYPES="long")
  def test_emulated_shr(self): self._test_shr()

  @given(ht.uint8, strat.sampled_from(integer_unary_operations))
  def test_uint8_unary(self, a, op): universal_test_unary(a, dtypes.uint8, op)

  @unittest.skipUnless(dtypes.uint16 in supported_dtypes, f"no uint16 on {Device.DEFAULT}")
  @given(ht.uint16, strat.sampled_from(integer_unary_operations))
  def test_uint16_unary(self, a, op): universal_test_unary(a, dtypes.uint16, op)

  @unittest.skipUnless(dtypes.uint32 in supported_dtypes, f"no uint32 on {Device.DEFAULT}")
  @given(ht.uint32, strat.sampled_from(integer_unary_operations))
  def test_uint32_unary(self, a, op): universal_test_unary(a, dtypes.uint32, op)

  @unittest.skipUnless(dtypes.uint64 in supported_dtypes, f"no uint64 on {Device.DEFAULT}")
  @given(ht.uint64, strat.sampled_from(integer_unary_operations))
  def test_uint64_unary(self, a, op): universal_test_unary(a, dtypes.uint64, op)

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

  @given(ht.int64, strat.sampled_from(integer_unary_operations))
  def test_int64_unary(self, a, op): universal_test_unary(a, dtypes.int64, op)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, PTXRenderer), "PTX does indexing math with longs")
  @given(ht.int64, strat.sampled_from(integer_unary_operations))
  @Context(EMULATED_DTYPES="long")
  def test_emulated_int64_unary(self, a, op): universal_test_unary(a, dtypes.int64, op)

  @given(ht.bool, ht.bool, strat.sampled_from(((operator.add, operator.add), (operator.mul, operator.mul))))
  def test_bool(self, a, b, op): universal_test(a, b, dtypes.bool, op)

  @given(ht.int32, ht.int32, ht.float32, strat.sampled_from(integer_binary_operations), strat.sampled_from(binary_operations))
  def test_int32_midcast_float(self, a, b, c, op1, op2): universal_test_midcast(a, b, c, op1, op2, dtypes.int32, dtypes.float32)

  # Metal and (MOCK)CUDA and HIP and NIR behave differently than numpy for overflows
  skip_overflow = ((DEV.interface.startswith("MOCK") and Device.DEFAULT in {"AMD", "NV", "CUDA"})
                   or isinstance(Device[Device.DEFAULT].renderer, NIRRenderer))
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
    if float_dtype not in supported_dtypes: float_dtype = dtypes.float32
    universal_test_cast(a, float_dtype, unsigned_dtype)

  @unittest.skip("relied on hacks")
  @given(strat.floats(width=32, min_value=256.0, max_value=65000.0, allow_subnormal=False),
         strat.sampled_from(dtypes_float), strat.sampled_from((dtypes.uint8, dtypes.uint16)))
  def test_float_cast_to_unsigned_overflow(self, a, float_dtype, unsigned_dtype):
    if float_dtype not in supported_dtypes: float_dtype = dtypes.float32
    universal_test_cast(a, float_dtype, unsigned_dtype)

  @unittest.skip("relied on hacks")
  @given(strat.floats(width=32, min_value=-65000.0, max_value=-1.0, allow_subnormal=False),
         strat.sampled_from(dtypes_float), strat.sampled_from((dtypes.uint8, dtypes.uint16)))
  def test_float_cast_to_unsigned_underflow(self, a, float_dtype, unsigned_dtype):
    if float_dtype not in supported_dtypes: float_dtype = dtypes.float32
    universal_test_cast(a, float_dtype, unsigned_dtype)

  def test_unsafe_cast_float_to_int(self):
    # the value is off the float32 grid but rounds in-range: the buffer and const-fold paths must agree
    # (out-of-range float->int cast stays undefined: hardware may saturate where the fold wraps)
    val = 2147483000.0
    t1 = Tensor([val], dtype=dtypes.float32).cast(dtypes.int32)
    t2 = Tensor(val, dtype=dtypes.float32).cast(dtypes.int32)
    np.testing.assert_equal(t1.item(), t2.item())

if __name__ == '__main__':
  unittest.main()
