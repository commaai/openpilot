import unittest, math
import numpy as np
import torch
from functools import partial
from tinygrad import nn, dtypes, Tensor, Device, TinyJit, Variable
from tinygrad.helpers import getenv, DEV, Context, OSX
from test.helpers import not_support_multi_device, needs_second_gpu
from hypothesis import given, settings, strategies as strat

settings.register_profile("my_profile", max_examples=200, deadline=None, derandomize=getenv("DERANDOMIZE_CI", False))
settings.load_profile("my_profile")

# https://gist.github.com/devries/11405101
def ksprob(a):
  fac, total, termbf = 2.0, 0.0, 0.0
  a2 = -2.0 * a * a
  for j in range(1, 101):
    term = fac * math.exp(a2 * j * j)
    total += term
    if math.fabs(term) <= 0.001 * termbf or math.fabs(term) <= 1e-8 * total:
      return total
    fac = -fac
    termbf = math.fabs(term)
  return 1.0

def kstest(l1, l2):
  n1, n2 = len(l1), len(l2)
  l1.sort()
  l2.sort()
  j1, j2, d, fn1, fn2 = 0, 0, 0.0, 0.0, 0.0
  while j1 < n1 and j2 < n2:
    d1, d2 = l1[j1], l2[j2]
    if d1 <= d2:
      fn1 = (float(j1) + 1.0) / float(n1)
      j1 += 1
    if d2 <= d1:
      fn2 = (float(j2) + 1.0) / float(n2)
      j2 += 1
    dtemp = math.fabs(fn2 - fn1)
    if dtemp > d:
      d = dtemp
  ne = float(n1 * n2) / float(n1 + n2)
  nesq = math.sqrt(ne)
  prob = ksprob((nesq + 0.12 + 0.11 / nesq) * d)
  return prob

def equal_distribution(tiny_func, torch_func=None, numpy_func=None, shape=(40, 43), alpha=0.04):
  Tensor.manual_seed(1337)
  torch.manual_seed(1337)
  np.random.seed(1337)
  assert not (torch_func is None and numpy_func is None), "no function to compare with"
  x1 = tiny_func(*shape).numpy().flatten()
  x2 = tiny_func(shape).numpy().flatten()
  if numpy_func is not None: y = numpy_func(shape).flatten()
  if torch_func is not None: z = torch_func(shape).numpy().flatten()
  return (numpy_func is None or (kstest(x1, y) >= alpha and kstest(x2, y) >= alpha)) and \
    (torch_func is None or (kstest(x1, z) >= alpha and kstest(x2, z) >= alpha))

class TestRandomness(unittest.TestCase):
  def test_rand(self):
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

  @unittest.skipIf(DEV.interface.startswith("MOCK") and Device.DEFAULT in {"NV", "CUDA"}, "gpuocelot doesn't support certain ops needed for threefry")
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

  @given(strat.sampled_from([dtypes.float, dtypes.float16, dtypes.bfloat16]))
  def test_randn_finite(self, default_float):
    if default_float not in Device[Device.DEFAULT].renderer.supported_dtypes(): return
    # low precision can result in inf from randn
    self.enterContext(Context(DEFAULT_FLOAT=default_float))
    t = Tensor.randn(64, 64)
    mx = t.max().numpy().item()
    mn = t.min().numpy().item()
    print(f"testing with {default_float=}")
    assert math.isfinite(mx), mx
    assert math.isfinite(mn), mn

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

  def test_three_lazy_rands_realized_one_at_a_time_are_distinct(self):
    Tensor.manual_seed(123)
    r1, r2, r3 = [Tensor.rand(4) for _ in range(3)]
    self.assertNotEqual(r1.tolist(), r2.tolist())
    self.assertNotEqual(r2.tolist(), r3.tolist())

  def test_randn(self):
    self.assertEqual(Tensor.randn(3,3,dtype=dtypes.half).dtype, dtypes.half)
    self.assertTrue(equal_distribution(Tensor.randn, torch.randn, lambda x: np.random.randn(*x)))

  def test_randint(self):
    self.assertTrue(equal_distribution(partial(Tensor.randint, low=-2, high=5),
                                       numpy_func=lambda x: np.random.randint(low=-2, high=5, size=x)))
    self.assertTrue(equal_distribution(partial(Tensor.randint, low=-2, high=5, dtype="int32"),
                                       numpy_func=lambda x: np.random.randint(low=-2, high=5, size=x)))
    self.assertTrue(Tensor.randint(1, device="CPU").device=="CPU")
    # check types of args
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=0.1, high=3)
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=0, high=3.5)
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=1, high=3, dtype="float")
    with self.assertRaises(TypeError): Tensor.randint((3, 4), low=0, high=3, dtype=dtypes.float32)
    # check low < high
    with self.assertRaises(ValueError): Tensor.randint((3, 4), low=10, high=5)
    with self.assertRaises(ValueError): Tensor.randint((3, 4), low=10, high=10)
    np.testing.assert_array_equal(Tensor.randint(16, low=5, high=6).numpy(), 5)

  def test_normal(self):
    self.assertTrue(equal_distribution(Tensor.normal, lambda x: torch.nn.init.normal_(torch.empty(x), mean=0, std=1),
                                                      lambda x: np.random.normal(loc=0, scale=1, size=x)))
    # check std >= 0
    with self.assertRaises(ValueError): Tensor.normal((3, 4), mean=0, std=-1)

  def test_uniform(self):
    self.assertTrue(equal_distribution(Tensor.uniform, lambda x: torch.nn.init.uniform_(torch.empty(x)), lambda x: np.random.uniform(size=x)))
    self.assertTrue(equal_distribution(partial(Tensor.uniform, low=-100, high=100, dtype=dtypes.int32),
                                       numpy_func=lambda x: np.random.randint(low=-100, high=100, size=x)))
    # check low < high
    with self.assertRaises(ValueError): Tensor.uniform((3, 4), low=5.0, high=3.0)
    with self.assertRaises(ValueError): Tensor.uniform((3, 4), low=1.0, high=1.0)

  def test_scaled_uniform(self):
    self.assertTrue(equal_distribution(Tensor.scaled_uniform, lambda x: torch.nn.init.uniform_(torch.empty(x), a=-1, b=1) / math.sqrt(math.prod(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) / math.sqrt(math.prod(x))))

  def test_glorot_uniform(self):
    self.assertTrue(equal_distribution(Tensor.glorot_uniform, lambda x: torch.nn.init.xavier_uniform_(torch.empty(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) * math.sqrt(6 / (x[0] + math.prod(x[1:])))))

  def test_kaiming_uniform(self):
    for shape in [(32, 16, 3, 3), (20, 44), (5, 15, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_uniform, lambda x: torch.nn.init.kaiming_uniform_(torch.empty(x)), shape=shape))

  def test_kaiming_normal(self):
    for shape in [(32, 16, 3, 3), (20, 44), (3, 15, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_normal, lambda x: torch.nn.init.kaiming_normal_(torch.empty(x)), shape=shape))

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "slow on WEBGPU")
  def test_multinomial(self):
    self.assertRaises(AssertionError, lambda: Tensor(2).multinomial(1, replacement=False))
    self.assertRaises(AssertionError, lambda: Tensor([1, 9]).multinomial(0, replacement=False))
    def _check_with_torch(w, num_samples, replacement):
      tiny_res = Tensor(w).multinomial(num_samples, replacement=replacement).realize()
      torch_res = torch.tensor(w).multinomial(num_samples, replacement=replacement)
      self.assertEqual(tiny_res.shape, torch_res.shape)
      if torch_res.ndim == 1:
        tiny_res = tiny_res.unsqueeze(0)
        torch_res = torch_res.unsqueeze(0)
      for i in range(torch_res.shape[0]):
        self.assertTrue(equal_distribution(lambda *_: tiny_res[i], lambda _: torch_res[i]))
    _check_with_torch(w=[0.231, 0., 1., 0.5], num_samples=300, replacement=True)
    _check_with_torch(w=[[0.2, 0.8]], num_samples=300, replacement=True)  # 2D but only 1 row
    _check_with_torch(w=[[0.453, 0., 1., 0.81], [0.1, 0.8, 0., 0.1]], num_samples=300, replacement=True)
    # no-replacement
    w = [0.1, 0.9]
    self.assertRaises(AssertionError, lambda: Tensor(w).multinomial(100, replacement=False))

    @TinyJit
    def sample_one(): return Tensor(w).multinomial(1, replacement=False).realize()

    tiny_samples = [sample_one().item() for _ in range(200)]
    torch_samples = [torch.tensor(w).multinomial(1, replacement=False).item() for _ in range(200)]
    self.assertTrue(equal_distribution(lambda *_: Tensor(tiny_samples), lambda _: torch.tensor(torch_samples)))

    w = list(range(32))
    s1 = Tensor(w).multinomial(5, replacement=False).numpy()
    self.assertEqual(len(set(s1.tolist())), 5)
    s2 = Tensor(w).multinomial(5, replacement=False).numpy()
    self.assertFalse(np.array_equal(s1, s2))
    full = Tensor(w).multinomial(len(w), replacement=False).numpy()
    self.assertEqual(sorted(full.tolist()), w)

    w = [0.1, 0.2, 0.3, 0.4]
    @TinyJit
    def sample_three(): return Tensor(w).multinomial(3, replacement=False).realize()

    tiny_draws = np.array([sample_three().numpy() for _ in range(200)])
    torch_draws = np.array([torch.tensor(w).multinomial(3, replacement=False).numpy() for _ in range(200)])
    for pos in range(3):
      self.assertTrue(equal_distribution(lambda *_: Tensor(tiny_draws[:, pos]), lambda _: torch.tensor(torch_draws[:, pos])))

  @unittest.skip("this test is flaky")
  def test_multinomial_counterexample(self):
    tiny_res = Tensor([0.3, 0.6, 0.1]).multinomial(4000, replacement=True)
    torch_res = torch.tensor([0.3, 0.6, 0.1]).multinomial(4000, replacement=True)
    self.assertTrue(equal_distribution(lambda *_: tiny_res, lambda _: torch_res))
    torch_res = torch.tensor([0.2, 0.7, 0.1]).multinomial(4000, replacement=True)
    self.assertFalse(equal_distribution(lambda *_: tiny_res, lambda _: torch_res))

  def test_conv2d_init(self):
    params = (32, 64, (3,3))
    assert equal_distribution(lambda *_: nn.Conv2d(*params).weight, lambda _: torch.nn.Conv2d(*params).weight.detach())
    assert equal_distribution(lambda *_: nn.Conv2d(*params).bias, lambda _: torch.nn.Conv2d(*params).bias.detach())

  def test_linear_init(self):
    params = (64, 256)
    assert equal_distribution(lambda *_: nn.Linear(*params).weight, lambda _: torch.nn.Linear(*params).weight.detach())
    assert equal_distribution(lambda *_: nn.Linear(*params).bias, lambda _: torch.nn.Linear(*params).bias.detach())

  def test_bn_init(self):
    params = (64,)
    assert equal_distribution(lambda *_: nn.BatchNorm2d(*params).weight, lambda _: torch.nn.BatchNorm2d(*params).weight.detach())
    assert equal_distribution(lambda *_: nn.BatchNorm2d(*params).bias, lambda _: torch.nn.BatchNorm2d(*params).bias.detach())

@unittest.skipIf(Device.DEFAULT == "WEBGPU" and not OSX, "WEBGPU Vulkan can only run kernels with up to 10 buffers")
class TestSample(unittest.TestCase):
  def test_sample(self):
    X = Tensor.rand(1000, 50).realize()
    BS = 16
    idxs = np.random.randint(0, X.shape[0], size=(BS))
    # this uncovered a bug with arg sort order
    batch = [Variable(f'idx{i}', 0, X.shape[0]-1).bind(s) for i,s in enumerate(idxs.tolist())]
    x = Tensor.cat(*[X.shrink(((batch[i], batch[i]+1), None)) for i in range(BS)])
    print(idxs)
    ret = x.numpy()
    base = X.numpy()[idxs]
    np.testing.assert_equal(ret, base)

if __name__ == '__main__':
  unittest.main()
