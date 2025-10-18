import unittest
from tinygrad import Tensor, Device, dtypes
from tinygrad.tensor import _to_np_dtype
from tinygrad.helpers import Context, getenv, CI, OSX
from test.test_schedule import check_schedule
from test.test_dtype_alu import ht, dtypes_float
from tinygrad.device import is_dtype_supported
import numpy as np
import math
from hypothesis import given, settings, strategies as strat

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

class TestTranscendentalMath(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.float64, Device.DEFAULT), f"no float64 on {Device.DEFAULT}")
  @unittest.skipIf(getenv("MOCKGPU") and Device.DEFAULT in {"NV", "CUDA"}, "crashed")
  @given(ht.float64, strat.sampled_from([(Tensor.exp, np.exp), (Tensor.log, np.log), (Tensor.sin, np.sin)]))
  def test_float64(self, x, op):
    if op[0] == Tensor.sin:
      # TODO: reduction does not work  # 536870912.125  # 2914593.01171875  # 134217728.03125  # 230581075.65625  # 139216373.71875
      if abs(x) > 100_000_000: return
    with Context(TRANSCENDENTAL=2), np.errstate(all='ignore'):
      np.testing.assert_allclose(op[0](Tensor([x], dtype=dtypes.float64)).numpy(),
                                 op[1](np.array([x], dtype=_to_np_dtype(dtypes.float64))),
                                 atol=3e-2, rtol=1e-5)  # sin can have bigger atol for very big x

  @unittest.skipIf(getenv("MOCKGPU") and Device.DEFAULT in {"NV", "CUDA"}, "crashed")
  @given(ht.float32, strat.sampled_from([(Tensor.exp, np.exp),(Tensor.log, np.log)] +
    ([(Tensor.sin, np.sin)] if is_dtype_supported(dtypes.ulong) else [])))
  def test_float32(self, x, op):
    # wrong nan behavior on Vulkan
    if (math.isnan(x) or (x < 0 and op[0] == Tensor.log)) and CI and Device.DEFAULT == "WEBGPU" and not OSX: return
    with Context(TRANSCENDENTAL=2), np.errstate(all='ignore'):
      np.testing.assert_allclose(op[0](Tensor([x], dtype=dtypes.float32)).numpy(),
                                 op[1](np.array([x], dtype=_to_np_dtype(dtypes.float32))),
                                 atol=2e-5, rtol=1e-5)

  @unittest.skipUnless(is_dtype_supported(dtypes.float16, Device.DEFAULT), f"no float16 on {Device.DEFAULT}")
  @given(ht.float16, strat.sampled_from([(Tensor.exp, np.exp),(Tensor.log, np.log)] +
    ([(Tensor.sin, np.sin)] if is_dtype_supported(dtypes.ulong) else [])))
  def test_float16(self, x, op):
    # wrong nan behavior on Vulkan
    if (math.isnan(x) or (x < 0 and op[0] == Tensor.log)) and CI and Device.DEFAULT == "WEBGPU" and not OSX: return
    with Context(TRANSCENDENTAL=2), np.errstate(all='ignore'):
      np.testing.assert_allclose(op[0](Tensor([x], dtype=dtypes.float16)).numpy(),
                                 op[1](np.array([x], dtype=_to_np_dtype(dtypes.float16))),
                                 atol=1e-2, rtol=5e-3)  # exp can have bigger rtol

  @given(strat.sampled_from([(dtypes.float64, 709.5), (dtypes.float32, 88.7), (dtypes.float16, 11)]))
  def test_exp_near_inf(self, dtype_x):
    # reordering compute might return inf
    dtype, x = dtype_x
    if not is_dtype_supported(dtype): return
    with Context(TRANSCENDENTAL=2):
      y = Tensor([x], dtype=dtype).exp().numpy()
      expected = np.exp(np.array([x], dtype=_to_np_dtype(dtype)))
      np.testing.assert_allclose(y, expected, rtol=5e-3)

class TestFromFuzzer(unittest.TestCase):
  @given(strat.sampled_from(dtypes_float))
  @unittest.skipUnless(is_dtype_supported(dtypes.ulong), "Needs ulong")
  def test_sin(self, dtype):
    if not is_dtype_supported(dtype): return
    if dtype == dtypes.float64:
      # crashes in CI CUDA
      if getenv("MOCKGPU") and Device.DEFAULT in {"NV", "CUDA"}: return
    def _test_value(n: float, unit: float=1.0):
      next_float = np.nextafter(1.0, 2.0, dtype=_to_np_dtype(dtype))
      ulp = next_float - 1.0
      ulp = unit * ulp
      with Context(TRANSCENDENTAL=2):
        np.testing.assert_allclose(Tensor([n], dtype=dtype).sin().numpy(), np.sin(np.array([n], dtype=_to_np_dtype(dtype))), atol=ulp, rtol=1e-5)
    _test_value(-35.0)
    _test_value(-25.0)
    _test_value(25.0)
    _test_value(30.0) # 30.0 == switch_over
    _test_value(35.0)
    _test_value(0.0)
    _test_value(np.pi / 2)
     # worst case of ulp 1.5
    _test_value(np.pi * 2, unit=1.5)

  @given(strat.sampled_from(dtypes_float))
  @unittest.skipIf(Device.DEFAULT == "WEBGPU" and CI, "Nan location mismatch on Vulkan, Metal works")
  def test_log2(self, dtype):
    if not is_dtype_supported(dtype): return
    if dtype == dtypes.float64:
      # crashes in CI CUDA
      if getenv("MOCKGPU") and Device.DEFAULT in {"NV", "CUDA"}: return
    def _test_value(n: float, unit: float=1.0):
      next_float = np.nextafter(1.0, 2.0, dtype=_to_np_dtype(dtype))
      ulp = next_float - 1.0
      ulp = unit * ulp
      with Context(TRANSCENDENTAL=2):
        np.testing.assert_allclose(Tensor([n], dtype=dtype).log2().numpy(), np.log2(np.array([n], dtype=_to_np_dtype(dtype))), atol=ulp, rtol=1e-5)
    fmin = np.finfo(_to_np_dtype(dtype)).tiny
    for scale in [1.0, 1e10, 1e20, 1e30]:
      _test_value(fmin * scale)
      _test_value(-fmin * scale)
    _test_value(0)
    _test_value(0.0000009)

class TestTranscendentalSchedule(unittest.TestCase):
  @unittest.skipUnless(is_dtype_supported(dtypes.ulong), "Needs ulong")
  def test_transcendental_sin_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.sin() + b.sin()
      c = c.sin()
      check_schedule(c, 1)

  def test_transcendental_log2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.log2() + b.log2()
      c = c.log2()
      check_schedule(c, 1)

  def test_transcendental_exp2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.exp2() + b.exp2()
      c = c.exp2()
      check_schedule(c, 1)

class TestTranscendentalVectorized(unittest.TestCase):
  def _vectorized_data(self, low, high, vec_size):
    np_data = np.linspace(low, high, num=(128 // vec_size) * vec_size, dtype=np.float32).reshape(-1, vec_size)
    data = Tensor(np_data, dtype=dtypes.float32.vec(vec_size))
    return data, np_data

  def _test_vectorized_op(self, fxn, np_fxn, data_range, vec_size, param_range=None):
    data, np_data = self._vectorized_data(data_range[0], data_range[1], vec_size)
    if param_range:
      param, np_param = self._vectorized_data(param_range[0], param_range[1], vec_size)
      out, np_out = fxn(data, param), np_fxn(np_data, np_param)
    else:
      out, np_out = fxn(data), np_fxn(np_data)
    np.testing.assert_allclose(out.numpy(), np_out, rtol=1e-4)

  def test_exp2_vectorized(self):
    for vec_size in [1,2,3,4,5,127,128]: self._test_vectorized_op(Tensor.exp2, np.exp2, (-100, 100), vec_size)

  def test_log2_vectorized(self):
    for vec_size in [1,2,3,4,5,127,128]: self._test_vectorized_op(Tensor.log2, np.log2, (0.001, 200), vec_size)

  @unittest.skipIf(getenv("DSP"), "requires int division")
  def test_sin_vectorized(self):
    for vec_size in [1,2,3,4,5,127,128]: self._test_vectorized_op(Tensor.sin, np.sin, (-100, 100), vec_size)

  def test_pow_vectorized(self):
    # np.pow returns nan for negative values raised to a non-integral power
    for vec_size in [1,2,3,4,5,127,128]: self._test_vectorized_op(Tensor.pow, np.pow, (0.001, 200), vec_size, param_range=(-10, 10))

  def test_sqrt_vectorized(self):
    for vec_size in [1,2,3,4,5,127,128]: self._test_vectorized_op(Tensor.sqrt, np.sqrt, (0, 100), vec_size)

if __name__ == '__main__':
  unittest.main()
