import unittest, math, struct, operator
from tinygrad.tensor import Tensor, dtypes
from tinygrad.dtype import DTYPES_DICT, truncate, float_to_fp16, float_to_bf16, _to_np_dtype, least_upper_dtype, least_upper_float
from tinygrad.device import is_dtype_supported

from tinygrad.helpers import getenv
from hypothesis import given, settings, strategies as strat
import numpy as np
import torch

settings.register_profile("my_profile", max_examples=50, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

core_dtypes = list(DTYPES_DICT.values())
dtype_ints = [dt for dt in core_dtypes if dtypes.is_int(dt) and is_dtype_supported(dt)]
dtype_floats = [dt for dt in core_dtypes if dtypes.is_float(dt) and is_dtype_supported(dt)]

FP8E4M3_MAX = 448.0
FP8E5M2_MAX = 57344.0

def u32_to_f32(u): return struct.unpack('f', struct.pack('I', u))[0]
def f32_to_u32(f): return struct.unpack('I', struct.pack('f', f))[0]

class TestHelpers(unittest.TestCase):
  signed_ints = (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64)
  uints = (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  floats = (dtypes.float16, dtypes.float32, dtypes.float64)

  @given(strat.sampled_from(signed_ints+uints), strat.integers(min_value=1, max_value=8))
  def test_is_int(self, dtype, amt):
    assert dtypes.is_int(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_float(dtype.vec(amt) if amt > 1 else dtype)

  @given(strat.sampled_from(uints), strat.integers(min_value=1, max_value=8))
  def test_is_unsigned_uints(self, dtype, amt):
    assert dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  @given(strat.sampled_from(signed_ints), strat.integers(min_value=1, max_value=8))
  def test_is_unsigned_signed_ints(self, dtype, amt):
    assert not dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  @given(strat.sampled_from(floats), strat.integers(min_value=1, max_value=8))
  def test_is_float(self, dtype, amt):
    assert dtypes.is_float(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_int(dtype.vec(amt) if amt > 1 else dtype)
    assert not dtypes.is_unsigned(dtype.vec(amt) if amt > 1 else dtype)

  def test_bf16_is_float(self):
    assert dtypes.is_float(dtypes.bfloat16)

  def test_fp8s_are_float(self):
    assert dtypes.is_float(dtypes.fp8e4m3)
    assert dtypes.is_float(dtypes.fp8e5m2)

  @given(strat.sampled_from([d for d in DTYPES_DICT.values() if dtypes.is_float(d) or dtypes.is_int(d)]), strat.integers(min_value=2, max_value=8))
  def test_scalar(self, dtype, amt):
    assert dtype.vec(amt).scalar() == dtype

  def test_from_py(self):
    assert dtypes.from_py(True) == dtypes.bool
    assert dtypes.from_py(2) == dtypes.default_int
    assert dtypes.from_py(3.0) == dtypes.default_float
    assert dtypes.from_py([]) == dtypes.default_float
    assert dtypes.from_py(()) == dtypes.default_float
    assert dtypes.from_py([True]) == dtypes.bool
    assert dtypes.from_py([True, 2]) == dtypes.default_int
    assert dtypes.from_py([True, 3.0]) == dtypes.default_float
    assert dtypes.from_py([2, 3.0]) == dtypes.default_float
    assert dtypes.from_py([True, 2, 3.0]) == dtypes.default_float
    with self.assertRaises(RuntimeError): dtypes.from_py(None)
    with self.assertRaises(RuntimeError): dtypes.from_py([None])
    with self.assertRaises(RuntimeError): dtypes.from_py({})
    with self.assertRaises(RuntimeError): dtypes.from_py(set())

  def test_dtype_range(self):
    for dt in core_dtypes:
      if dtypes.is_float(dt):
        np.testing.assert_equal(dtypes.min(dt), -math.inf)
        np.testing.assert_equal(dtypes.max(dt), math.inf)
        np.testing.assert_equal(dt.min, -math.inf)
        np.testing.assert_equal(dt.max, math.inf)
      elif dtypes.is_int(dt):
        info = np.iinfo(_to_np_dtype(dt))
        np.testing.assert_equal(dtypes.min(dt), info.min)
        np.testing.assert_equal(dtypes.max(dt), info.max)
        np.testing.assert_equal(dt.min, info.min)
        np.testing.assert_equal(dt.max, info.max)
      else:
        assert dt == dtypes.bool, dt
        np.testing.assert_equal(dtypes.min(dt), False)
        np.testing.assert_equal(dtypes.max(dt), True)
        np.testing.assert_equal(dt.min, False)
        np.testing.assert_equal(dt.max, True)

  def test_dtype_range_vec(self):
    for dt in core_dtypes:
      self.assertEqual(dt.min, dt.vec(4).min)
      self.assertEqual(dt.max, dt.vec(4).max)

  def test_float_to_fp16(self):
    self.assertEqual(float_to_fp16(1), 1)
    self.assertEqual(float_to_fp16(65504), 65504)
    self.assertEqual(float_to_fp16(65519.999), 65504)
    self.assertEqual(float_to_fp16(65520), math.inf)
    self.assertEqual(float_to_fp16(1e-8), 0.0)
    self.assertEqual(float_to_fp16(-65504), -65504)
    self.assertEqual(float_to_fp16(-65519.999), -65504)
    self.assertEqual(float_to_fp16(-65520), -math.inf)
    self.assertTrue(math.isnan(float_to_fp16(math.nan)))

  def test_float_to_bf16(self):
    max_bf16 = torch.finfo(torch.bfloat16).max
    for a in [1, 1.1, 1234, 23456, -777.777, max_bf16, max_bf16 * 1.00001, -max_bf16, -max_bf16 * 1.00001, math.inf, -math.inf]:
      self.assertEqual(float_to_bf16(a), torch.tensor([a], dtype=torch.bfloat16).item())
    self.assertTrue(math.isnan(float_to_bf16(math.nan)))

  def test_float_to_bf16_nan(self):
    patterns = [0x7FC00001, 0xFFC00001, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF]
    for u in patterns:
      x = u32_to_f32(u)
      y = float_to_bf16(x)
      t = torch.tensor([x], dtype=torch.bfloat16).item()
      self.assertTrue(math.isnan(y))
      self.assertTrue(math.isnan(t))

  def test_float_to_bf16_round(self):
    uppers = [0x3f800000, 0x41230000, 0xC1460000]
    for upper in uppers:
      base = upper & 0xFFFF0000
      base_f32 = u32_to_f32(base)
      base_f32_round_up = u32_to_f32(base + 0x00010000)

      x = u32_to_f32(base | 0x00007000)
      self.assertEqual(float_to_bf16(x), base_f32)
      self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32)

      x = u32_to_f32(base | 0x0000C000)
      self.assertEqual(float_to_bf16(x), base_f32_round_up)
      self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32_round_up)

      if ((upper >> 16) & 1) == 0:
        x = u32_to_f32(base | 0x00008000)
        self.assertEqual(float_to_bf16(x), base_f32)
        self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32)
      else:
        x = u32_to_f32(base | 0x00008000)
        self.assertEqual(float_to_bf16(x), base_f32_round_up)
        self.assertEqual(torch.tensor([x], dtype=torch.bfloat16).item(), base_f32_round_up)

  def test_float_to_bf16_boundary(self):
    base = 0x7F7F0000
    inf_u32 = 0x7F800000

    x = u32_to_f32(base | 0x00007FFF)
    self.assertEqual(f32_to_u32(float_to_bf16(x)), base)
    self.assertEqual(f32_to_u32(torch.tensor([x], dtype=torch.bfloat16).item()), base)

    x = u32_to_f32(base | 0x0000C000)
    self.assertEqual(f32_to_u32(float_to_bf16(x)), inf_u32)
    self.assertEqual(f32_to_u32(torch.tensor([x], dtype=torch.bfloat16).item()), inf_u32)

    x = u32_to_f32(base | 0x00008000)
    self.assertEqual(f32_to_u32(float_to_bf16(x)), inf_u32)
    self.assertEqual(f32_to_u32(torch.tensor([x], dtype=torch.bfloat16).item()), inf_u32)

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=True, allow_infinity=True))
  def test_truncate_fp8e4m3(self, x):
    if math.isnan(x): np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), x)
    elif math.isinf(x): np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), math.copysign(math.nan, x))
    elif x > FP8E4M3_MAX: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), FP8E4M3_MAX)
    elif x < -FP8E4M3_MAX: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), -FP8E4M3_MAX)
    else: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), torch.tensor(x, dtype=torch.float8_e4m3fn).float().item())

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=True, allow_infinity=True))
  def test_truncate_fp8e5m2(self, x):
    if math.isnan(x): np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), x)
    elif math.isinf(x): np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), x)
    elif x > FP8E5M2_MAX: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), FP8E5M2_MAX)
    elif x < -FP8E5M2_MAX: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), -FP8E5M2_MAX)
    else: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), torch.tensor(x, dtype=torch.float8_e5m2).float().item())

class TestTypePromotion(unittest.TestCase):
  @given(strat.sampled_from(core_dtypes))
  def test_self_promo_to_self(self, dtype):
    assert least_upper_dtype(dtype) == dtype
    assert least_upper_dtype(dtype, dtype) == dtype
    assert least_upper_dtype(dtype, dtype, dtype) == dtype

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_promo_resulted_higher_than_inputs(self, dtype1, dtype2):
    result = least_upper_dtype(dtype1, dtype2)
    assert not (result < dtype1) and not (result < dtype2)

  def test_dtype_promo(self):
    assert least_upper_dtype(dtypes.bool, dtypes.int8) == dtypes.int8
    assert least_upper_dtype(dtypes.int8, dtypes.uint8) == dtypes.int16
    assert least_upper_dtype(dtypes.uint8, dtypes.int16) == dtypes.int16
    assert least_upper_dtype(dtypes.int16, dtypes.uint16) == dtypes.int32
    assert least_upper_dtype(dtypes.uint16, dtypes.int32) == dtypes.int32
    assert least_upper_dtype(dtypes.int32, dtypes.uint32) == dtypes.int64
    assert least_upper_dtype(dtypes.uint32, dtypes.int64) == dtypes.int64
    assert least_upper_dtype(dtypes.int64, dtypes.uint64) == dtypes.uint64
    assert least_upper_dtype(dtypes.float16, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.float32, dtypes.float64) == dtypes.float64

    assert least_upper_dtype(dtypes.bool, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.bool, dtypes.float64) == dtypes.float64
    assert least_upper_dtype(dtypes.float16, dtypes.int64) == dtypes.float16
    assert least_upper_dtype(dtypes.float16, dtypes.uint64) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.fp8e5m2) == dtypes.half
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.bfloat16) == dtypes.bfloat16
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.bfloat16) == dtypes.bfloat16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.float16) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.float16) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.int64) == dtypes.fp8e4m3
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.uint64) == dtypes.fp8e4m3
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.int64) == dtypes.fp8e5m2
    assert least_upper_dtype(dtypes.fp8e5m2, dtypes.uint64) == dtypes.fp8e5m2

class TestTypeSpec(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

  def test_set_dtype_default(self):
    for default_int in [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]:
      dtypes.default_int = default_int
      assert dtypes.default_int == default_int

    for default_float in [*dtypes.fp8s, dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]:
      dtypes.default_float = default_float
      assert dtypes.default_float == default_float

  @given(strat.sampled_from(core_dtypes), strat.sampled_from([operator.gt, operator.ge, operator.le, operator.lt, operator.eq, operator.ne]))
  def test_bool_ops(self, dtype, op):
    assert op(Tensor.ones(4, 4, dtype=dtype), Tensor.ones(4, 4, dtype=dtype)).dtype == dtypes.bool

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_functions_return_index(self, dtype, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    assert Tensor([0, 1], dtype=dtype).argmax().dtype == dtypes.int32
    assert Tensor([0, 1], dtype=dtype).argmin().dtype == dtypes.int32
    assert Tensor([0, 1], dtype=dtype).multinomial().dtype == dtypes.int32

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(dtype_ints))
  def test_tensor_indexing_returns_same_dtype(self, data_dtype, indices_dtype):
    X_data =  Tensor.ones(60000, 1, 28, 28, dtype=data_dtype)
    indices =  Tensor.randint(512, high=X_data.shape[0]).cast(indices_dtype)
    assert X_data[indices].dtype == X_data.dtype

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(dtype_ints))
  def test_gather_returns_same_dtype(self, data_dtype, indices_dtype):
    X_data = Tensor([[1, 0], [0, 1]], dtype=data_dtype)
    indices = Tensor([[0, 0], [1, 0]], dtype=indices_dtype)
    assert X_data.gather(0, indices).dtype == X_data.dtype
    assert X_data.gather(1, indices).dtype == X_data.dtype

  @given(strat.sampled_from(dtype_floats), strat.sampled_from(dtype_floats))
  def test_attention_returns_same_dtype(self, data_dtype, default_float):
    dtypes.default_float = default_float
    query = Tensor.rand(32, 8, 128, 64, dtype=data_dtype)
    key = Tensor.rand(32, 8, 128, 64, dtype=data_dtype)
    value = Tensor.rand(32, 8, 128, 64, dtype=data_dtype)
    mask = (Tensor.rand(32, 8, 128, 128) < 0.5)
    assert query.scaled_dot_product_attention(key, value, is_causal=True).dtype == data_dtype
    assert query.scaled_dot_product_attention(key, value, is_causal=True, dropout_p=0.3).dtype == data_dtype
    assert query.scaled_dot_product_attention(key, value, is_causal=False).dtype == data_dtype
    assert query.scaled_dot_product_attention(key, value, attn_mask=mask).dtype == data_dtype

class TestAutoCastType(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

  @given(strat.sampled_from(dtype_floats), strat.sampled_from(dtype_floats))
  def test_least_upper_float_input_is_float(self, input_dtype, default_float):
    dtypes.default_float = default_float
    self.assertEqual(least_upper_float(input_dtype), input_dtype)

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_least_upper_float_input_is_int(self, input_dtype, default_float):
    dtypes.default_float = default_float
    self.assertEqual(least_upper_float(input_dtype), default_float)

  @given(strat.sampled_from(core_dtypes))
  def test_broadcast_scalar(self, dt):
    assert (Tensor.ones(4, 4, dtype=dt) + 2.3).dtype == (dt if dtypes.is_float(dt) else dtypes.default_float)
    assert (Tensor.ones(4, 4, dtype=dt) + 2).dtype == (dt if dtypes.is_float(dt) or dtypes.is_int(dt) else dtypes.default_int)
    assert (Tensor.ones(4, 4, dtype=dt) + True).dtype == dt

  @given(strat.sampled_from(dtype_floats))
  def test_int_div_int(self, default_float):
    dtypes.default_float = default_float
    self.assertEqual(Tensor([1]).div(Tensor([2])).dtype, default_float)

  def test_sum(self):
    assert (Tensor([0, 1], dtype=dtypes.bool)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int8)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int16)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int32)).sum().dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int64)).sum().dtype == dtypes.int64
    assert (Tensor([0, 1], dtype=dtypes.uint8)).sum().dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint16)).sum().dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint32)).sum().dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint64)).sum().dtype == dtypes.uint64
    assert (Tensor([0, 1], dtype=dtypes.fp8e4m3)).sum().dtype == dtypes.fp8e4m3
    assert (Tensor([0, 1], dtype=dtypes.fp8e5m2)).sum().dtype == dtypes.fp8e5m2
    assert (Tensor([0, 1], dtype=dtypes.float16)).sum().dtype == dtypes.float16
    assert (Tensor([0, 1], dtype=dtypes.bfloat16)).sum().dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).sum().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).sum().dtype == dtypes.float64

  def test_mean(self):
    assert (Tensor([0, 1], dtype=dtypes.bool)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int8)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int16)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int32)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.int64)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint8)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint16)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint32)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.uint64)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.fp8e4m3)).mean().dtype == dtypes.fp8e4m3
    assert (Tensor([0, 1], dtype=dtypes.fp8e5m2)).mean().dtype == dtypes.fp8e5m2
    assert (Tensor([0, 1], dtype=dtypes.float16)).mean().dtype == dtypes.float16
    assert (Tensor([0, 1], dtype=dtypes.bfloat16)).mean().dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).mean().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).mean().dtype == dtypes.float64

  def test_cumsum(self):
    assert (Tensor([0, 1], dtype=dtypes.bool)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int8)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int16)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int32)).cumsum(0).dtype == dtypes.int32
    assert (Tensor([0, 1], dtype=dtypes.int64)).cumsum(0).dtype == dtypes.int64
    assert (Tensor([0, 1], dtype=dtypes.uint8)).cumsum(0).dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint16)).cumsum(0).dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint32)).cumsum(0).dtype == dtypes.uint32
    assert (Tensor([0, 1], dtype=dtypes.uint64)).cumsum(0).dtype == dtypes.uint64
    assert (Tensor([0, 1], dtype=dtypes.fp8e4m3)).cumsum(0).dtype == dtypes.fp8e4m3
    assert (Tensor([0, 1], dtype=dtypes.fp8e5m2)).cumsum(0).dtype == dtypes.fp8e5m2
    assert (Tensor([0, 1], dtype=dtypes.float16)).cumsum(0).dtype == dtypes.float16
    assert (Tensor([0, 1], dtype=dtypes.bfloat16)).cumsum(0).dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).cumsum(0).dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).cumsum(0).dtype == dtypes.float64

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_matmul(self, dt1, dt2, acc_dt):
    t1 = Tensor([0, 1], dtype=dt1)
    t2 = Tensor([0, 1], dtype=dt2)
    self.assertEqual(t1.matmul(t2).dtype, least_upper_dtype(t1.dtype, t2.dtype))
    self.assertEqual(t1.matmul(t2, dtype=acc_dt).dtype, acc_dt)

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_linear(self, dt1, dt2, dt3, acc_dt):
    x = Tensor([0, 1], dtype=dt1)
    w = Tensor([0, 1], dtype=dt2)
    b = Tensor([0, 1], dtype=dt3)
    self.assertEqual(x.linear(w).dtype, least_upper_dtype(x.dtype, w.dtype))
    self.assertEqual(x.linear(w, b).dtype, least_upper_dtype(least_upper_dtype(x.dtype, w.dtype), b.dtype))
    self.assertEqual(x.linear(w, dtype=acc_dt).dtype, acc_dt)
    self.assertEqual(x.linear(w, b, dtype=acc_dt).dtype, acc_dt)

  @staticmethod
  def check_where_alternate_input_other(input_, other, data_type):
    assert (Tensor([True, False]).where(input_, other)).dtype == data_type
    assert (Tensor([True, False]).where(other, input_)).dtype == data_type

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_where_no_scalar(self, dt1, dt2):
    self.check_where_alternate_input_other(Tensor(2, dtype=dt1), Tensor(3, dtype=dt2), least_upper_dtype(dt1, dt2))

  @given(strat.sampled_from(core_dtypes))
  def test_where_one_scalar(self, dt):
    t = Tensor(2, dtype=dt)
    self.check_where_alternate_input_other(t, 3.2, (dt if dtypes.is_float(dt) else dtypes.default_float))
    self.check_where_alternate_input_other(t, 3, (dt if dtypes.is_float(dt) or dtypes.is_int(dt) else dtypes.default_int))
    self.check_where_alternate_input_other(t, True, dt)

  def test_where_two_scalars(self):
    self.check_where_alternate_input_other(3.1, 3.2, dtypes.default_float)
    self.check_where_alternate_input_other(3.1, 3, dtypes.default_float)
    self.check_where_alternate_input_other(3.1, True, dtypes.default_float)
    self.check_where_alternate_input_other(3, 2, dtypes.default_int)
    self.check_where_alternate_input_other(3, True, dtypes.default_int)
    self.check_where_alternate_input_other(False, True, dtypes.bool)

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_maximum(self, dt1, dt2):
    assert Tensor([0, 1, 2], dtype=dt1).maximum(Tensor([2, 0, 5], dtype=dt2)).dtype == least_upper_dtype(dt1, dt2)

  @given(strat.sampled_from(core_dtypes))
  def test_maximum_const(self, dt):
    assert Tensor([1, 2], dtype=dt).maximum(3.1).dtype == (dt if dtypes.is_float(dt) else dtypes.default_float)
    assert Tensor([1, 2], dtype=dt).maximum(3).dtype == (dt if dtypes.is_float(dt) or dtypes.is_int(dt) else dtypes.default_int)
    assert Tensor([1, 2], dtype=dt).maximum(True).dtype == dt

  def test_div(self):
    assert (Tensor([1, 2], dtype=dtypes.int32) / Tensor([2, 2], dtype=dtypes.int32)).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.int16) / Tensor([2, 2], dtype=dtypes.int32)).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.float32) / Tensor([2, 2], dtype=dtypes.float16)).dtype == dtypes.float32
    assert (Tensor([1, 2], dtype=dtypes.int32) / Tensor([2, 2], dtype=dtypes.float16)).dtype == dtypes.float16

  def test_div_const(self):
    assert (Tensor([1, 2], dtype=dtypes.int32) / 2).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.int32) / 2.0).dtype == dtypes.default_float
    assert (Tensor([1, 2], dtype=dtypes.float16) / 2).dtype == dtypes.float16
    assert (Tensor([1, 2], dtype=dtypes.float16) / 2.0).dtype == dtypes.float16

if __name__ == '__main__':
  unittest.main()
