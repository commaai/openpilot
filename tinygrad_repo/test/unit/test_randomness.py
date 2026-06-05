import unittest, math, torch
import numpy as np
from functools import partial
from tinygrad import nn, dtypes, Tensor, Device, TinyJit, Variable
from tinygrad.helpers import OSX

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

def normal_test(func, shape=(20, 45), alpha=0.05): return equal_distribution(func, numpy_func=lambda x: np.random.randn(*x), shape=shape, alpha=alpha)

class TestRandomness(unittest.TestCase):
  def test_randn(self):
    self.assertEqual(Tensor.randn(3,3,dtype=dtypes.half).dtype, dtypes.half)
    self.assertTrue(normal_test(Tensor.randn))
    self.assertTrue(equal_distribution(Tensor.randn, torch.randn, lambda x: np.random.randn(*x)))

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
    # check low < high
    with self.assertRaises(ValueError): Tensor.randint((3, 4), low=10, high=5)
    with self.assertRaises(ValueError): Tensor.randint((3, 4), low=10, high=10)
    np.testing.assert_array_equal(Tensor.randint(16, low=5, high=6).numpy(), 5)

  def test_normal(self):
    self.assertTrue(normal_test(Tensor.normal))
    self.assertTrue(equal_distribution(Tensor.normal, lambda x: torch.nn.init.normal_(torch.empty(x), mean=0, std=1),
                                                      lambda x: np.random.normal(loc=0, scale=1, size=x)))
    # check std >= 0
    with self.assertRaises(ValueError): Tensor.normal((3, 4), mean=0, std=-1)

  def test_uniform(self):
    self.assertFalse(normal_test(Tensor.uniform))
    self.assertTrue(equal_distribution(Tensor.uniform, lambda x: torch.nn.init.uniform_(torch.empty(x)), lambda x: np.random.uniform(size=x)))
    self.assertTrue(equal_distribution(partial(Tensor.uniform, low=-100, high=100, dtype=dtypes.int32),
                                       numpy_func=lambda x: np.random.randint(low=-100, high=100, size=x)))
    # check low < high
    with self.assertRaises(ValueError): Tensor.uniform((3, 4), low=5.0, high=3.0)
    with self.assertRaises(ValueError): Tensor.uniform((3, 4), low=1.0, high=1.0)

  def test_scaled_uniform(self):
    self.assertFalse(normal_test(Tensor.scaled_uniform))
    self.assertTrue(equal_distribution(Tensor.scaled_uniform, lambda x: torch.nn.init.uniform_(torch.empty(x), a=-1, b=1) / math.sqrt(math.prod(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) / math.sqrt(math.prod(x))))

  def test_glorot_uniform(self):
    self.assertFalse(normal_test(Tensor.glorot_uniform))
    self.assertTrue(equal_distribution(Tensor.glorot_uniform, lambda x: torch.nn.init.xavier_uniform_(torch.empty(x)),
                                                              lambda x: np.random.uniform(-1, 1, size=x) * math.sqrt(6 / (x[0] + math.prod(x[1:])))))

  def test_kaiming_uniform(self):
    for shape in [(32, 16, 3, 3), (20, 44), (5, 15, 35)]:
      self.assertTrue(equal_distribution(Tensor.kaiming_uniform, lambda x: torch.nn.init.kaiming_uniform_(torch.empty(x)), shape=shape))

  def test_kaiming_normal(self):
    for shape in [(32, 16, 3, 3), (20, 44), (3, 15, 35)]:
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
    _check_with_torch(w=[0.231, 0., 1., 0.5], num_samples=300, replacement=True)
    _check_with_torch(w=[[0.2, 0.8]], num_samples=300, replacement=True)  # 2D but only 1 row
    _check_with_torch(w=[[0.453, 0., 1., 0.81], [0.1, 0.8, 0., 0.1]], num_samples=300, replacement=True)
    # no-replacement
    w = [0.1, 0.9]
    self.assertRaises(AssertionError, lambda: Tensor(w).multinomial(100, replacement=False))

    @TinyJit
    def sample_one(): return Tensor(w).multinomial(1, replacement=False).realize()

    tiny_samples = [sample_one().item() for _ in range(400)]
    torch_samples = [torch.tensor(w).multinomial(1, replacement=False).item() for _ in range(400)]
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

    tiny_draws = np.array([sample_three().numpy() for _ in range(400)])
    torch_draws = np.array([torch.tensor(w).multinomial(3, replacement=False).numpy() for _ in range(400)])
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

# TODO: still fails with MAX_KERNEL_BUFFERS
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

if __name__ == "__main__":
  unittest.main()
