import unittest, math, operator, subprocess
from tinygrad.tensor import Tensor, dtypes, Device
from tinygrad.dtype import DType, DTYPES_DICT, truncate, truncate_fp16, truncate_bf16, _to_np_dtype, least_upper_dtype, least_upper_float
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import getenv, CI, DEBUG
from hypothesis import given, settings, strategies as strat
import numpy as np
import torch
import ml_dtypes

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

core_dtypes = list(DTYPES_DICT.values())
dtype_ints = [dt for dt in core_dtypes if dtypes.is_int(dt) and is_dtype_supported(dt)]
dtype_floats = [dt for dt in core_dtypes if dtypes.is_float(dt) and is_dtype_supported(dt)]

FP8E4M3_MAX = 448.0
FP8E5M2_MAX = 57344.0

def _assert_eq(tensor:Tensor, target_dtype:DType, target, tol_target_dtype:float=1e-7):
  if DEBUG >= 2: print(tensor.numpy())
  try:
    assert tensor.dtype == target_dtype
    np.testing.assert_allclose(tensor.numpy(), target, rtol={dtypes.float16:1e-3, dtypes.bfloat16:1e-2}.get(target_dtype, tol_target_dtype))
  except AssertionError as e:
    raise AssertionError(f"\ntensor {tensor.numpy()} dtype {tensor.dtype} does not match target {target} with dtype {target_dtype}") from e

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

  def test_truncate_fp16(self):
    self.assertEqual(truncate_fp16(1), 1)
    self.assertEqual(truncate_fp16(65504), 65504)
    self.assertEqual(truncate_fp16(65519.999), 65504)
    self.assertEqual(truncate_fp16(65520), math.inf)

  def test_truncate_bf16(self):
    self.assertEqual(truncate_bf16(1), 1)
    self.assertAlmostEqual(truncate_bf16(1.1), 1.09375, places=7)
    for a in [1234, 23456, -777.777]:
      self.assertEqual(truncate_bf16(a), torch.tensor([a], dtype=torch.bfloat16).item())
    # TODO: torch bfloat 1.1 gives 1.1015625 instead of 1.09375
    max_bf16 = torch.finfo(torch.bfloat16).max
    self.assertEqual(truncate_bf16(max_bf16), max_bf16)
    self.assertEqual(truncate_bf16(min_bf16:=-max_bf16), min_bf16)
    self.assertEqual(truncate_bf16(max_bf16 * 1.00001), math.inf)
    self.assertEqual(truncate_bf16(min_bf16 * 1.00001), -math.inf)

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=True, allow_infinity=True))
  def test_truncate_fp8e4m3(self, x):
    if x > FP8E4M3_MAX: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), FP8E4M3_MAX)
    elif x < -FP8E4M3_MAX: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), -FP8E4M3_MAX)
    else: np.testing.assert_equal(truncate[dtypes.fp8e4m3](x), ml_dtypes.float8_e4m3fn(x))

  @given(strat.floats(width=32, allow_subnormal=True, allow_nan=True, allow_infinity=True))
  def test_truncate_fp8e5m2(self, x):
    if x > FP8E5M2_MAX: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), FP8E5M2_MAX)
    elif x < -FP8E5M2_MAX: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), -FP8E5M2_MAX)
    else: np.testing.assert_equal(truncate[dtypes.fp8e5m2](x), ml_dtypes.float8_e5m2(x))

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

  @unittest.skip("this test is slow and spawning whole pythons")
  def test_env_set_default_float(self):
    # check default
    subprocess.run(['python3 -c "from tinygrad import dtypes; assert dtypes.default_float == dtypes.float"'],
                    shell=True, check=True)
    # check change
    subprocess.run(['DEFAULT_FLOAT=HALF python3 -c "from tinygrad import dtypes; assert dtypes.default_float == dtypes.half"'],
                    shell=True, check=True)
    # check invalid
    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.run(['DEFAULT_FLOAT=INT32 python3 -c "from tinygrad import dtypes"'],
                      shell=True, check=True)

    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.run(['DEFAULT_FLOAT=TYPO python3 -c "from tinygrad import dtypes"'],
                      shell=True, check=True)

  @unittest.skipUnless(is_dtype_supported(dtypes.int8), f"no int8 on {Device.DEFAULT}")
  def test_dtype_str_arg(self):
    n = np.random.normal(0, 1, (10, 10)).astype(np.float32)
    tested = 0
    for dtype_str, dtype in [
      ("bool", dtypes.bool), ("int8", dtypes.int8), ("int", dtypes.int), ("uint32", dtypes.uint32), ("float32", dtypes.float32)]:
      np.testing.assert_equal(Tensor(n, dtype=dtype_str).numpy(), Tensor(n, dtype=dtype).numpy())
      np.testing.assert_equal(Tensor(n).cast(dtype_str).numpy(), Tensor(n).cast(dtype).numpy())
      if dtype.itemsize == 4:
        np.testing.assert_equal(Tensor(n).bitcast(dtype_str).numpy(), Tensor(n).bitcast(dtype).numpy())
        tested += 1
    assert tested == 3

    with self.assertRaises(AttributeError): Tensor([1, 2, 3], dtype="nonexistdtype")
    with self.assertRaises(AttributeError): Tensor([1, 2, 3], dtype="")

    np.testing.assert_equal(Tensor(n).sum(dtype="int16").numpy(), Tensor(n).sum(dtype=dtypes.int16).numpy())

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_creation(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    _assert_eq(Tensor(True), dtypes.bool, True)
    _assert_eq(Tensor(None), dtypes.default_float, [])
    _assert_eq(Tensor(2), dtypes.default_int, 2)
    _assert_eq(Tensor(2.34), dtypes.default_float, 2.34)
    _assert_eq(Tensor([]), dtypes.default_float, [])
    _assert_eq(Tensor([1]), dtypes.default_int, [1])
    _assert_eq(Tensor([1.1]), dtypes.default_float, [1.1])

    _assert_eq(Tensor.eye(0), dtypes.default_float, np.eye(0))
    _assert_eq(Tensor.eye(3), dtypes.default_float, np.eye(3))
    if is_dtype_supported(dtypes.int64):
      _assert_eq(Tensor.eye(3, dtype=dtypes.int64), dtypes.int64, np.eye(3))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.eye(3, dtype=dtypes.float16), dtypes.float16, np.eye(3))

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_full(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float

    _assert_eq(Tensor.zeros((2, 3)), dtypes.default_float, np.zeros((2, 3)))
    if is_dtype_supported(dtypes.int64):
      _assert_eq(Tensor.zeros((2, 3), dtype=dtypes.int64), dtypes.int64, np.zeros((2, 3)))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.zeros((2, 3), dtype=dtypes.float16), dtypes.float16, np.zeros((2, 3)))

    _assert_eq(Tensor.ones((2, 3)), dtypes.default_float, np.ones((2, 3)))
    if is_dtype_supported(dtypes.int64):
      _assert_eq(Tensor.ones((2, 3), dtype=dtypes.int64), dtypes.int64, np.ones((2, 3)))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.ones((2, 3), dtype=dtypes.float16), dtypes.float16, np.ones((2, 3)))

    _assert_eq(Tensor.full((2, 3), 3.0), dtypes.default_float, np.full((2, 3), 3.0))
    _assert_eq(Tensor.full((2, 3), 3), dtypes.default_int, np.full((2, 3), 3))
    _assert_eq(Tensor.full((2, 3), True), dtypes.bool, np.full((2, 3), True))
    if is_dtype_supported(dtypes.int64):
      _assert_eq(Tensor.full((2, 3), 3, dtype=dtypes.int64), dtypes.int64, np.full((2, 3), 3))
      _assert_eq(Tensor.full((2, 3), 3.0, dtype=dtypes.int64), dtypes.int64, np.full((2, 3), 3))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.full((2, 3), 3, dtype=dtypes.float16), dtypes.float16, np.full((2, 3), 3))
      _assert_eq(Tensor.full((2, 3), 3.0, dtype=dtypes.float16), dtypes.float16, np.full((2, 3), 3))

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_reduce_0d_default(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float
    _assert_eq(Tensor.ones((2,3,0)).sum(2), dtypes.default_float, np.zeros((2, 3)))
    # TODO: what should this one be?
    # _assert_eq(Tensor.ones((2,3,0), dtype=dtypes.default_int).sum(2), dtypes.default_int, np.zeros((2, 3)))
    _assert_eq(Tensor.ones((2,3,0), dtype=dtypes.int32).sum(2), dtypes.int32, np.zeros((2, 3)))

  @given(strat.sampled_from(dtype_ints), strat.sampled_from(dtype_floats))
  def test_arange(self, default_int, default_float):
    dtypes.default_int, dtypes.default_float = default_int, default_float

    _assert_eq(Tensor.arange(5), dtypes.default_int, np.arange(5))
    _assert_eq(Tensor.arange(120), dtypes.default_int, np.arange(120))
    _assert_eq(Tensor.arange(5.0), dtypes.default_float, np.arange(5))
    if is_dtype_supported(dtypes.int16):
      _assert_eq(Tensor.arange(5, dtype=dtypes.int16), dtypes.int16, np.arange(5))
    if is_dtype_supported(dtypes.int64):
      _assert_eq(Tensor.arange(5, dtype=dtypes.int64), dtypes.int64, np.arange(5))
    if is_dtype_supported(dtypes.float16):
      _assert_eq(Tensor.arange(5, dtype=dtypes.float16), dtypes.float16, np.arange(5))
    _assert_eq(Tensor.arange(3, 9, 0.7), dtypes.default_float, np.arange(3, 9, 0.7), 1e-6 if Device.DEFAULT == "WEBGPU" else 1e-7)
    _assert_eq(Tensor.arange(3, 8.5, 3), dtypes.default_float, np.arange(3, 8.5, 3))
    # stop-start and step have different signs
    _assert_eq(Tensor.arange(3, 5, -2), dtypes.default_int, np.arange(3, 5, -2))
    _assert_eq(Tensor.arange(5.0, 3.0), dtypes.default_float, np.arange(5.0, 3.0))

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
    # similar to jax but we don't use weak type
    assert least_upper_dtype(dtypes.int64, dtypes.uint64) == dtypes.float16
    assert least_upper_dtype(dtypes.float16, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.float32, dtypes.float64) == dtypes.float64

    assert least_upper_dtype(dtypes.bool, dtypes.float32) == dtypes.float32
    assert least_upper_dtype(dtypes.bool, dtypes.float64) == dtypes.float64
    assert least_upper_dtype(dtypes.float16, dtypes.int64) == dtypes.float16
    assert least_upper_dtype(dtypes.float16, dtypes.uint64) == dtypes.float16
    assert least_upper_dtype(dtypes.fp8e4m3, dtypes.fp8e5m2) == dtypes.half

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

  @given(strat.sampled_from([d for d in core_dtypes if dtypes.is_int(d) and is_dtype_supported(d)]))
  def test_int_to_float_unary_func(self, dtype):
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
      # float16 can have larger precision errors
      np.testing.assert_allclose(func(Tensor(a, dtype=dtype)).numpy(), func(torch.tensor(a)), rtol=1e-3, atol=1e-3)

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
    assert (Tensor([0, 1], dtype=dtypes.float16)).sum().dtype == dtypes.float16
    #assert (Tensor([0, 1], dtype=dtypes.bfloat16)).sum().dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).sum().dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).sum().dtype == dtypes.float64

  @unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16")
  def test_sum_dtype_arg(self):
    t = Tensor([40000, 40000], dtype=dtypes.float16)
    # default float16 sum returns in float16, overflowed in this case
    assert t.sum().dtype == dtypes.float16
    assert math.isinf(t.sum().numpy().item())
    # specifiying dtype and it's not downcasted
    assert t.sum(dtype=dtypes.float32).dtype == dtypes.float32
    np.testing.assert_allclose(t.sum(dtype=dtypes.float32).numpy(), 80000)

  def test_prod_dtype_arg(self):
    t = Tensor([100, 200], dtype=dtypes.int32)
    assert t.prod().dtype == dtypes.int32
    np.testing.assert_allclose(t.prod().numpy(), 20000)
    assert t.prod(dtype=dtypes.float32).dtype == dtypes.float32
    np.testing.assert_allclose(t.prod(dtype=dtypes.float32).numpy(), 20000)

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
    assert (Tensor([0, 1], dtype=dtypes.float16)).mean().dtype == dtypes.float16
    #assert (Tensor([0, 1], dtype=dtypes.bfloat16)).mean().dtype == dtypes.bfloat16
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
    assert (Tensor([0, 1], dtype=dtypes.float16)).cumsum(0).dtype == dtypes.float16
    #assert (Tensor([0, 1], dtype=dtypes.bfloat16)).cumsum(0).dtype == dtypes.bfloat16
    assert (Tensor([0, 1], dtype=dtypes.float32)).cumsum(0).dtype == dtypes.float32
    assert (Tensor([0, 1], dtype=dtypes.float64)).cumsum(0).dtype == dtypes.float64

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_matmul(self, dt1, dt2, acc_dt):
    t1 = Tensor([0, 1], dtype=dt1)
    t2 = Tensor([0, 1], dtype=dt2)
    self.assertEqual(t1.matmul(t2).dtype, least_upper_dtype(t1.dtype, t2.dtype))
    # if dtype is specified, return in dtype
    self.assertEqual(t1.matmul(t2, dtype=acc_dt).dtype, acc_dt)

  @given(strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes), strat.sampled_from(core_dtypes))
  def test_linear(self, dt1, dt2, dt3, acc_dt):
    x = Tensor([0, 1], dtype=dt1)
    w = Tensor([0, 1], dtype=dt2)
    b = Tensor([0, 1], dtype=dt3)
    self.assertEqual(x.linear(w).dtype, least_upper_dtype(x.dtype, w.dtype))
    self.assertEqual(x.linear(w, b).dtype, least_upper_dtype(least_upper_dtype(x.dtype, w.dtype), b.dtype))
    # if dtype is specified, return in dtype
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

  def test_gradient_dtype(self):
    old_default_float = dtypes.default_float

    for default_dtype in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]:
      if not is_dtype_supported(default_dtype): continue
      dtypes.default_float = default_dtype
      for dtype in [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]:
        if not is_dtype_supported(dtype): continue
        if DEBUG >= 2:
          print(f"testing {default_dtype=}, {dtype=}")
        a = Tensor([1, 2, 3], dtype=dtype, requires_grad=True)
        b = (a * 5).sum()
        b.backward()  # if there is dtype mismatch, lazy should assert
        assert a.grad.dtype == a.dtype
        np.testing.assert_allclose(a.grad.numpy(), [5, 5, 5])

    dtypes.default_float = old_default_float

  @unittest.skipIf(CI, "TODO: broken RuntimeError: Attempting to relocate against an undefined symbol 'fmaxf'")
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_backward_sum_acc_dtype(self):
    # test acc of sum in the backward is upcasted to float
    t = Tensor([5, -5], dtype=dtypes.half, requires_grad=True)
    t.reshape(2, 1).expand(2, 10001).max().backward()
    np.testing.assert_allclose(t.grad.numpy(), [1, 0])

  @unittest.skipIf(Device.DEFAULT == "PYTHON", "very slow")
  @unittest.skipIf(CI and Device.DEFAULT == "AMD", "very slow")
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Binding size is larger than the maximum storage buffer binding size")
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_mean_half_precision_underflow(self):
    N = 10000
    x = 0.001
    t = Tensor([[x]], dtype=dtypes.half, requires_grad=True).expand(N, N).contiguous()
    np.testing.assert_allclose(t.mean(axis=1).numpy(), np.array([x] * N, dtype=np.float16), rtol=1e-3)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_mean_half_precision_overflow(self):
    N = 256
    t = Tensor([60000] * N*N, dtype=dtypes.half, requires_grad=True).reshape(N, N)
    np.testing.assert_allclose(t.mean().numpy(), 60000)
    t.square().mean().backward()
    np.testing.assert_allclose(t.grad.numpy().flatten(), [60000 * 2 / (N*N)] * N*N)

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Precision error")
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_softmax_dtype(self):
    data = [1, 2, 3]
    t = Tensor(data, dtype=dtypes.half)
    tt = torch.tensor(data, dtype=torch.half)

    out = t.softmax(0)
    self.assertEqual(out.dtype, dtypes.half)
    np.testing.assert_allclose(out.numpy(), tt.softmax(0).numpy(), rtol=1e-3)
    out = t.softmax(0, dtype=dtypes.float)
    self.assertEqual(out.dtype, dtypes.float)
    np.testing.assert_allclose(out.numpy(), tt.softmax(0, dtype=torch.float).numpy(), rtol=1e-3)
    out = t.log_softmax(0)
    self.assertEqual(out.dtype, dtypes.half)
    np.testing.assert_allclose(out.numpy(), tt.log_softmax(0).numpy(), rtol=1e-3)
    out = t.log_softmax(0, dtype=dtypes.float)
    self.assertEqual(out.dtype, dtypes.float)
    np.testing.assert_allclose(out.numpy(), tt.log_softmax(0, dtype=torch.float).numpy(), rtol=1e-3)