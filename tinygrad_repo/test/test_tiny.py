# basic self-contained tests of the external functionality of tinygrad
import unittest, random
from tinygrad import Tensor, Context, Variable, TinyJit, dtypes, Device, nn
from tinygrad.helpers import IMAGE, CI

class TestTiny(unittest.TestCase):

  # *** basic functionality ***

  def test_plus(self):
    out = Tensor([1.,2,3]) + Tensor([4.,5,6])
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_plus_int(self):
    out = Tensor([1,2,3], dtype=dtypes.int) + Tensor([4,5,6], dtype=dtypes.int)
    self.assertListEqual(out.tolist(), [5, 7, 9])

  def test_plus_big(self):
    out = Tensor.ones(16).contiguous() + Tensor.ones(16).contiguous()
    self.assertListEqual(out.tolist(), [2]*16)

  def test_cat(self):
    out = Tensor.cat(Tensor.ones(8).contiguous(), Tensor.ones(8).contiguous())
    self.assertListEqual(out.tolist(), [1]*16)

  def test_sum(self):
    out = Tensor.ones(256).contiguous().sum()
    self.assertEqual(out.item(), 256)

  def test_gemm(self, N=64, out_dtype=dtypes.float):
    a = Tensor.ones(N,N).contiguous()
    b = Tensor.eye(N).contiguous()
    self.assertListEqual((out:=a@b).flatten().tolist(), [1.0]*(N*N))
    if IMAGE < 2: self.assertEqual(out.dtype, out_dtype)

  # *** randomness ***

  def test_random(self):
    out = Tensor.rand(10)
    for x in out.tolist():
      self.assertGreaterEqual(x, 0.0)
      self.assertLessEqual(x, 1.0)

  # *** JIT (for Python speed) ***

  def test_jit(self):
    cnt = 0
    random.seed(0)
    def new_rand_list(ln=10): return [random.randint(0, 100000) for _ in range(ln)]

    @TinyJit
    def fxn(a,b) -> Tensor:
      nonlocal cnt
      cnt += 1
      return a+b

    for _ in range(3):
      la,lb = new_rand_list(), new_rand_list()
      fa,fb = Tensor(la), Tensor(lb)
      ret = fxn(fa, fb)
      # math is correct
      self.assertListEqual(ret.tolist(), [a+b for a,b in zip(la, lb)])

    # function is only called twice
    self.assertEqual(cnt, 2)

  # *** BEAM (for Kernel speed) ***

  def test_beam(self):
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1): self.test_plus()

  # *** symbolic (to allow less recompilation) ***

  def test_symbolic(self):
    i = Variable('i', 1, 10)
    for s in [2,5]:
      ret = Tensor.ones(s).contiguous().reshape(i.bind(s)) + 1
      self.assertListEqual(ret.reshape(s).tolist(), [2.0]*s)

  def test_symbolic_reduce(self):
    i = Variable('i', 1, 10)
    for s in [2,5]:
      ret = Tensor.ones(s).contiguous().reshape(i.bind(s)).sum()
      self.assertEqual(ret.item(), s)

  # *** a model ***

  # TODO: this is failing because of how swizzling rewrites the ShapeTracker of the final STORE
  @unittest.skipIf(IMAGE>0 or (CI and Device.DEFAULT == "DSP"), "failing because of make things that can't be images not images")
  def test_mnist_model(self):
    layers = [
      nn.Conv2d(1, 32, 5), Tensor.relu,
      nn.Conv2d(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.relu,
      nn.Conv2d(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

    # replace random weights with ones
    for p in nn.state.get_parameters(layers): p.replace(Tensor.ones_like(p).contiguous()).realize()

    # run model inference
    probs = Tensor.rand(1, 1, 28, 28).sequential(layers).tolist()
    self.assertEqual(len(probs[0]), 10)

  # *** image ***

  @unittest.skipIf(Device.DEFAULT != "GPU", "image only supported on GPU")
  def test_image(self):
    with Context(IMAGE=2): self.test_gemm(N=4, out_dtype=dtypes.imagef((4, 1, 4)))

  def test_beam_image(self):
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1): self.test_image()

if __name__ == '__main__':
  unittest.main()

