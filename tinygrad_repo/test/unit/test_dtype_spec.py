import unittest, math, subprocess
from tinygrad.tensor import Tensor, dtypes, Device
from tinygrad.dtype import DType, DTYPES_DICT
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import getenv, DEBUG
from test.helpers import slow
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

def _assert_eq(tensor:Tensor, target_dtype:DType, target, tol_target_dtype:float=1e-7):
  if DEBUG >= 2: print(tensor.numpy())
  try:
    assert tensor.dtype == target_dtype
    np.testing.assert_allclose(tensor.numpy(), target, rtol={dtypes.float16:1e-3, dtypes.bfloat16:1e-2,
      dtypes.fp8e4m3:1e-1, dtypes.fp8e5m2:5e-1}.get(target_dtype, tol_target_dtype))

  except AssertionError as e:
    raise AssertionError(f"\ntensor {tensor.numpy()} dtype {tensor.dtype} does not match target {target} with dtype {target_dtype}") from e

class TestTypeSpec(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

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

class TestAutoCastType(unittest.TestCase):
  def setUp(self):
    self.old_default_int, self.old_default_float = dtypes.default_int, dtypes.default_float
  def tearDown(self):
    dtypes.default_int, dtypes.default_float = self.old_default_int, self.old_default_float

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

  def test_gradient_dtype(self):
    old_default_float = dtypes.default_float

    for default_dtype in dtypes.floats:
      if not is_dtype_supported(default_dtype): continue
      dtypes.default_float = default_dtype
      for dtype in  dtypes.floats:
        if not is_dtype_supported(dtype): continue
        if DEBUG >= 2:
          print(f"testing {default_dtype=}, {dtype=}")
        a = Tensor([1, 2, 3], dtype=dtype, requires_grad=True)
        b = (a * 5).sum()
        b.backward()  # if there is dtype mismatch, lazy should assert
        assert a.grad.dtype == a.dtype
        np.testing.assert_allclose(a.grad.numpy(), [5, 5, 5])

    dtypes.default_float = old_default_float

  @unittest.skipIf(Device.DEFAULT == "PYTHON", "very slow")
  @slow
  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "Binding size is larger than the maximum storage buffer binding size")
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_mean_half_precision_underflow(self):
    N = 10000
    x = 0.001
    t = Tensor([[x]], dtype=dtypes.half, requires_grad=True).expand(N, N).contiguous()
    np.testing.assert_allclose(t.mean(axis=1).numpy(), np.array([x] * N, dtype=np.float16), rtol=1e-3)

  @unittest.skip("this test only works with SPLIT_REDUCEOP=1")
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
