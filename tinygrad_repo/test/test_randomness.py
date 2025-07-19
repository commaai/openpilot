import unittest, math
from functools import partial

import numpy as np
import torch
from tinygrad import nn, dtypes, Tensor, Device, TinyJit
from tinygrad.helpers import getenv, CI
from tinygrad.device import is_dtype_supported
from tinygrad.engine.realize import lower_schedule, CompiledRunner
from hypothesis import given, settings, strategies as strat
from test.helpers import not_support_multi_device

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

def normal_test(func, shape=(20, 23), alpha=0.05): return equal_distribution(func, numpy_func=lambda x: np.random.randn(*x), shape=shape, alpha=alpha)

class TestRandomness(unittest.TestCase):
  def test_rand(self):
    self.assertFalse(normal_test(Tensor.rand))
    self.assertTrue(equal_distribution(Tensor.rand, torch.rand, lambda x: np.random.rand(*x)))

  @unittest.skipUnless(is_dtype_supported(dtypes.float16) and is_dtype_supported(dtypes.ulong), "need float16 and ulong support")
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

  @unittest.skipIf(getenv("PTX"), "fails with PTX")
  def test_threefry_doesnt_use_long(self):
    for (_,ei) in lower_schedule(Tensor.rand(20).schedule()):
      if isinstance(ei.prg, CompiledRunner):
        for u in ei.prg.p.uops:
          self.assertNotIn(u.dtype, {dtypes.long, dtypes.ulong}, msg=f"long found in {ei.prg.p.name}")

  def test_threefry_against_reference_full(self):
    Tensor.manual_seed(1337)

    # reference generated using
    """
    key0 = 1337
    key1 = int.from_bytes(hashlib.sha256(int(0).to_bytes(4)).digest(), "big") & 0xffffffff
    values = jax.extend.random.threefry_2x32((np.uint32(key1), np.uint32(key0)), np.arange(20, dtype=np.uint32))
    values = (values >> (32 - 23)) | np.array(1, dtype=np.float32).view(np.uint32)
    values =  values.view(np.float32) - 1
    print(f"[{', '.join(f'{v}' for v in values)}]")
    """
    jr = np.array([0.9073467254638672, 0.8235964775085449, 0.6872662305831909, 0.9920015335083008, 0.4941047430038452,
                   0.3108327388763428, 0.09639489650726318, 0.004686474800109863, 0.8435229063034058, 0.824237585067749,
                   0.5873836278915405, 0.4232727289199829, 0.2530076503753662, 0.40300023555755615, 0.03966474533081055,
                   0.27904558181762695, 0.9150195121765137, 0.48057758808135986, 0.23821306228637695, 0.7676635980606079], dtype=np.float32)
    r = Tensor.rand(20).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

    # next 20, np.arange(20, 40, dtype=np.uint32)
    jr = np.array([0.7444133758544922, 0.7713677883148193, 0.8233780860900879, 0.43871235847473145, 0.517757773399353,
                   0.6437174081802368, 0.967403769493103, 0.26167726516723633, 0.6825339794158936, 0.14966607093811035,
                   0.28920769691467285, 0.017063498497009277, 0.2627382278442383, 0.9525482654571533, 0.9351049661636353,
                   0.43904995918273926, 0.043945908546447754, 0.6616791486740112, 0.6667773723602295, 0.5228077173233032], dtype=np.float32)
    r = Tensor.rand(20).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

    # next 10, np.arange(40, 50, dtype=np.uint32)
    jr = np.array([0.9614430665969849, 0.059279561042785645, 0.01909029483795166, 0.47882091999053955, 0.9677121639251709,
                   0.36863112449645996, 0.3102607727050781, 0.06608951091766357, 0.35329878330230713, 0.26518797874450684], dtype=np.float32)
    r = Tensor.rand(10).numpy()
    np.testing.assert_allclose(r, jr, atol=1e-5, rtol=1e-5)

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

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_threefry_same_kernels(self):
    Tensor.manual_seed(0)

    Tensor.rand(1).realize()

    s = Tensor.rand(20).schedule()
    s2 = Tensor.rand(20).schedule()

    assert len(s) == len(s2), f"{len(s)} != {len(s2)}"
    for x,y in zip(s, s2):
      if not (x.ast == y.ast):
        print(f"{x.ast} != {y.ast}")

    Tensor.rand(1, device=f"{Device.DEFAULT}:1").realize()

    s3 = Tensor.rand(20, device=f"{Device.DEFAULT}:1").schedule()
    s4 = Tensor.rand(20, device=f"{Device.DEFAULT}:1").schedule()

    assert len(s3) == len(s4), f"{len(s3)} != {len(s4)}"
    assert len(s2) == len(s4), f"{len(s)} != {len(s3)}"
    for x,y in zip(s3, s4):
      if not (x.ast == y.ast):
        print(f"{x.ast} != {y.ast}")

  @unittest.skipUnless(is_dtype_supported(dtypes.bfloat16), "need bfloat16 support")
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

  def test_randn(self):
    self.assertEqual(Tensor.randn(3,3,dtype=dtypes.half).dtype, dtypes.half)
    self.assertTrue(normal_test(Tensor.randn))
    self.assertTrue(equal_distribution(Tensor.randn, torch.randn, lambda x: np.random.randn(*x)))

  def test_randn_device(self):
    self.assertEqual(Tensor.randn(3,3,device="CPU").device, "CPU")

  @given(strat.sampled_from([dtypes.float, dtypes.float16, dtypes.bfloat16]))
  @unittest.skipIf(Device.DEFAULT in ["HSA", "AMD"], "bfloat16 local buffer broken in HSA")
  def test_randn_finite(self, default_float):
    if not is_dtype_supported(default_float): return
    old_default_float = dtypes.default_float
    # low precision can result in inf from randn
    dtypes.default_float = default_float
    t = Tensor.randn(256, 256)
    mx = t.max().numpy().item()
    mn = t.min().numpy().item()
    print(f"testing with {default_float=}")
    assert math.isfinite(mx), mx
    assert math.isfinite(mn), mn
    dtypes.default_float = old_default_float

  def test_randint(self):
    self.assertFalse(normal_test(Tensor.randint))
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

  def test_normal(self):
    self.assertTrue(normal_test(Tensor.normal))
    self.assertTrue(equal_distribution(Tensor.normal, lambda x: torch.nn.init.normal_(torch.empty(x), mean=0, std=1),
                                                      lambda x: np.random.normal(loc=0, scale=1, size=x)))

  def test_uniform(self):
    self.assertFalse(normal_test(Tensor.uniform))
    self.assertTrue(equal_distribution(Tensor.uniform, lambda x: torch.nn.init.uniform_(torch.empty(x)), lambda x: np.random.uniform(size=x)))
    self.assertTrue(equal_distribution(partial(Tensor.uniform, low=-100, high=100, dtype=dtypes.int32),
                                       numpy_func=lambda x: np.random.randint(low=-100, high=100, size=x)))

  def test_scaled_uniform(self):
    self.assertFalse(normal_test(Tensor.scaled_uniform))
    self.assertTrue(equal_distribution(Tensor.scaled_uniform, lambda x: torch.nn.init.uniform_(torch.empty(x), a=-1, b=1) / math.sqrt(math.prod(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) / math.sqrt(math.prod(x))))

  def test_glorot_uniform(self):
    self.assertFalse(normal_test(Tensor.glorot_uniform))
    self.assertTrue(equal_distribution(Tensor.glorot_uniform, lambda x: torch.nn.init.xavier_uniform_(torch.empty(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) * math.sqrt(6 / (x[0] + math.prod(x[1:])))))

  def test_kaiming_uniform(self):
    for shape in [(32, 128, 3, 3), (80, 44), (3, 55, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_uniform, lambda x: torch.nn.init.kaiming_uniform_(torch.empty(x)), shape=shape))

  def test_kaiming_normal(self):
    for shape in [(32, 128, 3, 3), (80, 44), (3, 55, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_normal, lambda x: torch.nn.init.kaiming_normal_(torch.empty(x)), shape=shape))

  def test_multinomial(self):
    self.assertRaises(AssertionError, lambda: Tensor(2).multinomial(1, replacement=False))
    self.assertRaises(AssertionError, lambda: Tensor([1, 9]).multinomial(0, replacement=False))
    def _check_with_torch(w, num_samples, replacement):
      tiny_res = Tensor(w).multinomial(num_samples, replacement=replacement)
      torch_res = torch.tensor(w).multinomial(num_samples, replacement=replacement)
      self.assertEqual(tiny_res.shape, torch_res.shape)
      if torch_res.ndim == 1:
        tiny_res = tiny_res.unsqueeze(0)
        torch_res = torch_res.unsqueeze(0)
      for i in range(torch_res.shape[0]):
        self.assertTrue(equal_distribution(lambda *_: tiny_res[i], lambda _: torch_res[i]))
    _check_with_torch(w=[0.231, 0., 1., 0.5], num_samples=2000, replacement=True)
    _check_with_torch(w=[[0.2, 0.8]], num_samples=2000, replacement=True)  # 2D but only 1 row
    _check_with_torch(w=[[0.453, 0., 1., 0.81], [0.1, 0.8, 0., 0.1]], num_samples=2000, replacement=True)
    # no-replacement isn't supported, unless taking only one sample
    w = [0.1, 0.9]
    self.assertRaises(AssertionError, lambda: Tensor(w).multinomial(100, replacement=False))

    @TinyJit
    def sample_one(): return Tensor(w).multinomial(1, replacement=False).realize()

    tiny_samples = [sample_one().item() for _ in range(1000)]
    torch_samples = [torch.tensor(w).multinomial(1, replacement=False).item() for _ in range(1000)]
    self.assertTrue(equal_distribution(lambda *_: Tensor(tiny_samples), lambda _: torch.tensor(torch_samples)))

  def test_multinomial_counterexample(self):
    tiny_res = Tensor([0.3, 0.6, 0.1]).multinomial(4000, replacement=True)
    torch_res = torch.tensor([0.3, 0.6, 0.1]).multinomial(4000, replacement=True)
    self.assertTrue(equal_distribution(lambda *_: tiny_res, lambda _: torch_res))
    torch_res = torch.tensor([0.2, 0.7, 0.1]).multinomial(4000, replacement=True)
    self.assertFalse(equal_distribution(lambda *_: tiny_res, lambda _: torch_res))

  def test_conv2d_init(self):
    params = (128, 256, (3,3))
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

if __name__ == "__main__":
  unittest.main()
