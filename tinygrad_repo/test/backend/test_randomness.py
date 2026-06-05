import unittest, math

from tinygrad import dtypes, Tensor, Device
from tinygrad.helpers import getenv
from tinygrad.codegen import to_program

from tinygrad.uop.ops import Ops
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.nir import NIRRenderer
from tinygrad.renderer.isa.x86 import X86Renderer
from test.helpers import not_support_multi_device, needs_second_gpu, CI
from test.unit.test_randomness import equal_distribution, normal_test

import numpy as np
import torch
from hypothesis import given, settings, strategies as strat

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

class TestRandomness(unittest.TestCase):
  def test_rand(self):
    self.assertFalse(normal_test(Tensor.rand))
    self.assertTrue(equal_distribution(Tensor.rand, torch.rand, lambda x: np.random.rand(*x)))

  def test_rand_is_lazy(self):
    Tensor.manual_seed(0)
    r1 = Tensor.rand(10)
    self.assertFalse(r1.uop.is_realized, "rand should be lazy - tensor should not be realized")
    counter = Tensor._device_rng_counters[Device.DEFAULT]
    self.assertFalse(counter.uop.is_realized, "rand should be lazy - counter should not be realized")
    # second rand triggers assign path
    r2 = Tensor.rand(10)
    self.assertFalse(r2.uop.is_realized, "rand should be lazy - tensor should not be realized after second rand")
    self.assertFalse(counter.uop.is_realized, "rand should be lazy - counter should not be realized after second rand")
    Tensor.realize(r1, r2)
    self.assertTrue(r1.uop.is_realized, "tensor should be realized after .realize()")
    self.assertTrue(r2.uop.is_realized, "tensor should be realized after .realize()")

  @unittest.skipUnless(dtypes.float16 in Device[Device.DEFAULT].renderer.supported_dtypes(), "need float16 support")
  def test_rand_float16(self):
    N = 128
    x = Tensor.rand((2, N, N), dtype=dtypes.float16)
    assert x.dtype == dtypes.float16
    nx = x.numpy()
    # seed dependant, check output range is [0, 1)
    assert nx[nx == 1].size == 0
    assert nx[nx == 0].size > 0
    equal_distribution(lambda *x: Tensor.rand(*x, dtype=dtypes.float16), torch.rand, lambda x: np.random.rand(*x), shape=(2, N, N))

  @unittest.skipIf(CI and Device.DEFAULT in {"NV", "CUDA"}, "gpuocelot doesn't support certain ops needed for threefry")
  def test_threefry_against_reference(self):
    Tensor.manual_seed(1337)

    # reference generated using
    """
    key0 = 1337
    key1 = 0
    values = jax.extend.random.threefry_2x32((np.uint32(key1), np.uint32(key0)), np.arange(20, dtype=np.uint32))
    print(f"[{', '.join(f'{v}' for v in values)}]")
    """
    jr = np.array([2221762175, 1752107825, 653745012, 1967534793, 1395205442, 3840423848, 2159346757,
                   603508235, 3319473678, 3363866483, 3544324138, 1436466838, 2169858556, 2570072943,
                   2387150698, 3678370550, 2911697663, 403244401, 2560861638, 1692360114])

    counts = Tensor.arange(20, dtype=dtypes.uint32)
    counts0, counts1 = counts.chunk(2)
    r = Tensor._threefry_random_bits(Tensor([0, 1337], dtype='uint32'), counts0, counts1).numpy()

    np.testing.assert_allclose(jr, r)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, (NIRRenderer, PTXRenderer)), "PTX and NIR use pointer arithmetic")
  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, X86Renderer), "X86 callee saved registers have ulong dtype")
  def test_threefry_doesnt_use_long(self):
    linear = Tensor.rand(20).schedule_linear()
    for call in linear.src:
      ast = call.src[0]
      if ast.op is Ops.SINK:
        prg = to_program(ast, renderer=Device[Device.DEFAULT].renderer)
        for u in tuple(prg.src[2].src):
          self.assertNotIn(u.dtype, {dtypes.long, dtypes.ulong}, msg=f"long found in {prg.arg.name}")

  def test_threefry_against_reference_full(self):
    Tensor.manual_seed(1337)

    # reference generated using
    """
    key0 = 1337
    key1 = int.from_bytes(hashlib.sha256(int(0).to_bytes(4)).digest(), "big") & 0xffffffff
    # derive new key for the counter offset (c_low=0, c_high=0 for first call)
    new_key_values = jax.extend.random.threefry_2x32((np.uint32(key1), np.uint32(key0)), np.array([0, 0], dtype=np.uint32))
    new_key = (np.uint32(new_key_values[0]), np.uint32(new_key_values[1]))
    values = jax.extend.random.threefry_2x32(new_key, np.arange(20, dtype=np.uint32))
    values = (values >> (32 - 23)) | np.array(1, dtype=np.float32).view(np.uint32)
    values = values.view(np.float32) - 1
    print(f"[{', '.join(f'{v}' for v in values)}]")
    """
    jr = np.array([0.45735931396484375, 0.6311527490615845, 0.15571284294128418, 0.8149417638778687, 0.7862188816070557,
                   0.8008807897567749, 0.568588376045227, 0.9852620363235474, 0.42314577102661133, 0.9811755418777466,
                   0.38059568405151367, 0.09186363220214844, 0.9497315883636475, 0.5826880931854248, 0.3796330690383911,
                   0.5610522031784058, 0.16122901439666748, 0.3732343912124634, 0.9795231819152832, 0.3280656337738037], dtype=np.float32)
    r = Tensor.rand(20).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

    # next 20 (c_low=20, c_high=0)
    jr = np.array([0.09199333190917969, 0.9130761623382568, 0.7048608064651489, 0.22254979610443115, 0.0014830827713012695,
                   0.37023448944091797, 0.7790107727050781, 0.7484984397888184, 0.7524604797363281, 0.19875383377075195,
                   0.48537540435791016, 0.10002851486206055, 0.5369305610656738, 0.3294715881347656, 0.5246957540512085,
                   0.7659651041030884, 0.7949080467224121, 0.34988296031951904, 0.9798505306243896, 0.2599533796310425], dtype=np.float32)
    r = Tensor.rand(20).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

    # next 10 (c_low=40, c_high=0)
    jr = np.array([0.3198714256286621, 0.7984923124313354, 0.320881724357605, 0.4716068506240845, 0.7323365211486816,
                   0.9663800001144409, 0.13873648643493652, 0.16062307357788086, 0.49300849437713623, 0.10077548027038574], dtype=np.float32)
    r = Tensor.rand(10).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

  @needs_second_gpu
  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_threefry_tensors_cnt(self):
    Tensor.manual_seed(1337)

    Tensor.rand(20).realize()

    assert len(Tensor._device_rng_counters) == 1
    assert len(Tensor._device_seeds) == 1

    Tensor.rand(20, device=f"{Device.DEFAULT}:1").realize()

    assert len(Tensor._device_rng_counters) == 2
    assert len(Tensor._device_seeds) == 2

    Tensor.manual_seed(2)

    assert len(Tensor._device_rng_counters) == 0
    assert len(Tensor._device_seeds) == 0

  @needs_second_gpu
  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_threefry_same_kernels(self):
    Tensor.manual_seed(0)

    Tensor.rand(1).realize()

    s = Tensor.rand(20).schedule_linear().src
    s2 = Tensor.rand(20).schedule_linear().src

    assert len(s) == len(s2), f"{len(s)} != {len(s2)}"
    for x,y in zip(s, s2):
      if not (x.src[0] == y.src[0]):
        print(f"{x.src[0]} != {y.src[0]}")

    Tensor.rand(1, device=f"{Device.DEFAULT}:1").realize()

    s3 = Tensor.rand(20, device=f"{Device.DEFAULT}:1").schedule_linear().src
    s4 = Tensor.rand(20, device=f"{Device.DEFAULT}:1").schedule_linear().src

    assert len(s3) == len(s4), f"{len(s3)} != {len(s4)}"
    assert len(s2) == len(s4), f"{len(s)} != {len(s3)}"
    for x,y in zip(s3, s4):
      if not (x.src[0] == y.src[0]):
        print(f"{x.src[0]} != {y.src[0]}")

  @unittest.skipUnless(dtypes.bfloat16 in Device[Device.DEFAULT].renderer.supported_dtypes(), "need bfloat16 support")
  def test_rand_bfloat16(self):
    N = 128
    x = Tensor.rand((2, N, N), dtype=dtypes.bfloat16)
    assert x.dtype == dtypes.bfloat16
    nx = x.numpy()
    assert nx[nx == 1].size == 0
    assert nx[nx == 0].size > 0
    equal_distribution(lambda *x: Tensor.rand(*x, dtype=dtypes.bfloat16).float(), torch.rand, lambda x: np.random.rand(*x), shape=(2, N, N))

  def test_rand_like(self):
    empty = Tensor.empty((80, 44))
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_randn_like(self):
    empty = Tensor.empty((80, 44))
    rand = Tensor.randn_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_zero_shape(self):
    empty = Tensor.empty(0, 20)
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_more_dims(self):
    empty = Tensor.empty((1, 2, 3, 4, 5, 6))
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

  def test_rand_like_dtype(self):
    empty = Tensor.empty((80, 44), dtype=dtypes.float16)
    rand = Tensor.rand_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

    empty = Tensor.empty((80, 44))
    rand = Tensor.rand_like(empty, dtype=dtypes.float16)
    assert rand.shape == empty.shape
    assert rand.dtype == dtypes.float16
    assert rand.device == empty.device

  def test_randn_like_dtype(self):
    empty = Tensor.empty((80, 44), dtype=dtypes.float16)
    rand = Tensor.randn_like(empty)
    assert rand.shape == empty.shape
    assert rand.dtype == empty.dtype
    assert rand.device == empty.device

    empty = Tensor.empty((80, 44))
    rand = Tensor.randn_like(empty, dtype=dtypes.float16)
    assert rand.shape == empty.shape
    assert rand.dtype == dtypes.float16
    assert rand.device == empty.device

  def test_randn_device(self):
    self.assertEqual(Tensor.randn(3,3,device="CPU").device, "CPU")

  @given(strat.sampled_from([dtypes.float, dtypes.float16, dtypes.bfloat16]))
  def test_randn_finite(self, default_float):
    if default_float not in Device[Device.DEFAULT].renderer.supported_dtypes(): return
    old_default_float = dtypes.default_float
    # low precision can result in inf from randn
    dtypes.default_float = default_float
    t = Tensor.randn(64, 64)
    mx = t.max().numpy().item()
    mn = t.min().numpy().item()
    print(f"testing with {default_float=}")
    assert math.isfinite(mx), mx
    assert math.isfinite(mn), mn
    dtypes.default_float = old_default_float

  def test_random_counter_overflow(self):
    device = Device.DEFAULT
    Tensor.manual_seed(1337)
    Tensor.rand(1).realize()

    Tensor._device_rng_counters[device].assign(Tensor([dtypes.uint32.max - 5, 0], device=device, dtype=dtypes.uint32)).realize()

    Tensor.rand(10).realize()
    c = Tensor._device_rng_counters[device].numpy()
    np.testing.assert_allclose(c, [4, 1])

    Tensor.rand(10).realize()
    c = Tensor._device_rng_counters[device].numpy()
    np.testing.assert_allclose(c, [14, 1])

if __name__ == "__main__":
  unittest.main()
