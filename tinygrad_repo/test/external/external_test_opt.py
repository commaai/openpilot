#!/usr/bin/env python
import gc, unittest
import numpy as np
import torch

from tinygrad import GlobalCounters, Tensor, Device
from tinygrad.helpers import getenv, Context
from tinygrad.nn.state import get_parameters
from tinygrad.engine.realize import capturing
from tinygrad.tensor import _to_np_dtype

class CLCache:
  def __init__(self, allowed=None, strict=False, preclear=True, var_vals=None):
    self.allowed, self.strict, self.preclear, self.var_vals = allowed, strict, preclear, var_vals if var_vals is not None else {}
    self.count = 0
  def add(self, ei): self.count += 1
  def __enter__(self):
    if self.preclear:
      gc.collect()
      for x in [x for x in gc.get_objects() if isinstance(x, Tensor)]:
        x.realize()
      GlobalCounters.reset()
    capturing.append(self)
    print("cache: entering")
    return self
  def __exit__(self, _type, value, traceback):
    capturing.clear()
    print(f"cache: exiting with size {self.count}", f"allowed {self.allowed}" if self.allowed is not None else "")
    if self.allowed is not None:
      assert self.count == self.allowed, f"{self.count} != {self.allowed}"

from extra.models.convnext import ConvNeXt
from extra.models.efficientnet import EfficientNet
from extra.models.resnet import ResNet18
from extra.models.vit import ViT

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestInferenceMinKernels(unittest.TestCase):
  def setUp(self):
    self.training_old = Tensor.training
    Tensor.training = False
  def tearDown(self):
    Tensor.training = self.training_old

  def test_convnext(self):
    model = ConvNeXt()
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=_to_np_dtype(p.dtype)))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(143):
      model(img).realize()

  def test_enet(self):
    model = EfficientNet(getenv("ENET_NUM", 0), has_se=False)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=_to_np_dtype(p.dtype)))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(51):
      model.forward(img).realize()

  def test_enet_se(self):
    model = EfficientNet(getenv("ENET_NUM", 0), has_se=True)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=_to_np_dtype(p.dtype)))
    img = Tensor.randn(1, 3, 224, 224)
    # TODO: this seems very high
    with CLCache(115):
      model.forward(img).realize()

  def test_resnet(self):
    model = ResNet18()
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=_to_np_dtype(p.dtype)))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(23):
      model.forward(img).realize()

  def test_vit(self):
    model = ViT(embed_dim=192, num_heads=3)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=_to_np_dtype(p.dtype)))
    img = Tensor.randn(1, 3, 224, 224)
    with CLCache(209) as cache: # NOTE: this is way too high
      out = model.forward(img)
      assert cache.count == 0, "ViT prerealized?"
      out.realize()

  @unittest.skip("llama is fp16 but CI does not have fp16")
  def test_llama(self):
    from examples.llama import Transformer
    args_tiny = {"dim": 512, "hidden_dim": 1024, "n_heads": 8, "n_layers": 4, "norm_eps": 1e-05, "vocab_size": 1000}
    model = Transformer(**args_tiny)
    for p in get_parameters(model): p.assign(np.zeros(p.shape, dtype=_to_np_dtype(p.dtype)))
    inp = Tensor([[1,2,3,4]])
    with CLCache(100):
      model(inp, 0).realize()

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOptBinOp(unittest.TestCase):
  def _test_no_binop_rerun(self, f1, f2=None, allowed=1):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache() as cache:
      c = f1(a, b)
      if f2 is not None: d = f2(a, b)
      c.realize()
      if f2 is not None: d.realize()
      assert cache.count == allowed, "binop was rerun!"
    if f2 is not None: np.testing.assert_allclose(c.numpy().ravel(), d.numpy().ravel(), rtol=1e-3, atol=1e-5)

  def test_no_binop_rerun(self): return self._test_no_binop_rerun(lambda a,b: a*b, lambda a,b: (a*b).reshape(16, 16, 1))
  def test_no_binop_rerun_alt(self): return self._test_no_binop_rerun(lambda a,b: (a*b).reshape(16, 16, 1), lambda a,b: a*b)
  def test_no_binop_rerun_reduce_broadcast(self):
    return self._test_no_binop_rerun(lambda a,b: a.sum()+b, lambda a,b: a.sum().reshape(1,1)+b, allowed=2)

  @unittest.skip("this test started failing with the new change, based movementop issue")
  def test_no_binop_rerun_transposed(self): return self._test_no_binop_rerun(lambda a,b: (a.T*b.T).T, lambda a,b: a*b)
  def test_no_binop_rerun_mid_reshape(self): return self._test_no_binop_rerun(lambda a,b: (a*b).reshape(256)+a.reshape(256))

  # currently non working tests
  # def test_no_binop_rerun_preshape(self): return self._test_no_binop_rerun(lambda a,b: a.reshape(16, 16, 1)*b.reshape(16, 16, 1), lambda a,b: a*b)
  #def test_no_binop_rerun_reduce(self): return self._test_no_binop_rerun(lambda a,b: (a*b).sum(), lambda a,b: (a*b).reshape(16, 16, 1).sum())
  #def test_no_binop_rerun_reduce_alt(self): return self._test_no_binop_rerun(lambda a,b: a.sum(1)+b[0], lambda a,b: a.sum(1).reshape(1,16)+b[0])

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOptReduceLoop(unittest.TestCase):
  def test_loop_left(self):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache() as cache:
      t = a.sum(0)
      b = t.reshape(16,1).expand(16,16).sum(0)
      c = (t+b)
      c.realize()
      assert cache.count == 2, "loop left fusion broken"

  def test_loop_right(self):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache() as cache:
      t = a.sum(0)
      b = t.reshape(16,1).expand(16,16).sum(0)
      c = (b+t)
      c.realize()
      assert cache.count == 2, "loop right fusion broken"

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOptWChild(unittest.TestCase):
  @unittest.skip("this no longer happens, use realize")
  def test_unrealized_child(self):
    a = Tensor.randn(16, 16)
    b = Tensor.randn(16, 16)
    with CLCache() as cache:
      c = (a*b).sum()
      d = c+1
      e = c+2 # noqa: F841
      d.realize()
      assert cache.count == 2, "don't fuse if you have children"

@unittest.skipUnless(Device.DEFAULT == "GPU", "Not Implemented")
class TestOpt(unittest.TestCase):
  def test_muladd(self):
    a,b,c = [Tensor.randn(2,2).realize() for _ in range(3)]
    na,nb,nc = a.numpy(),b.numpy(),c.numpy()
    with CLCache(allowed=1):
      d = a * b + c
      d.realize()
    np.testing.assert_allclose(d.numpy(), na*nb+nc, rtol=1e-5, atol=1e-7)

  def test_permute_was_pushed(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache(2):
      c = a.sum(2)
      d = c.permute(1,0).contiguous()
      d.realize()
    np.testing.assert_allclose(a.numpy().sum(2).transpose(1,0), d.numpy(), rtol=1e-3, atol=1e-5)

  def test_permute_was_pushed_through_contract_reshape(self):
    a = Tensor.randn(4, 4, 4, 4, 4)
    with CLCache(2):
      c = a.sum(-1)
      d = c.reshape(16,16).permute(1,0).contiguous()
      d.realize()
    np.testing.assert_allclose(a.numpy().sum(-1).reshape(16,16).transpose(1,0), d.numpy(), rtol=1e-3, atol=1e-5)

  def test_permute_was_pushed_through_contractw1s_reshape(self):
    a = Tensor.randn(4, 4, 4, 4, 4)
    with CLCache(2):
      c = a.sum(-1)
      d = c.reshape(16,1,16).permute(2,1,0).contiguous()
      d.realize()
    np.testing.assert_allclose(a.numpy().sum(-1).reshape(16,1,16).transpose(2,1,0), d.numpy(), rtol=1e-3, atol=1e-5)

  def test_permute_was_pushed_through_expand_reshape(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache(2):
      c = a.sum(2)
      d = c.reshape(4,4,4,4).permute(2,3,0,1).contiguous()
      d.realize()
    np.testing.assert_allclose(a.numpy().sum(2).transpose(1,0).reshape(4,4,4,4), d.numpy(), rtol=1e-3, atol=1e-5)

  def test_no_reduceop_rerun(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache() as cache:
      c = a.sum(2)
      d = a.sum(2).permute(1,0)
      c.realize()
      d.realize()
      cache_len = cache.count
    np.testing.assert_allclose(c.numpy().transpose(1,0), d.numpy(), rtol=1e-3, atol=1e-5)
    assert cache_len == 1, "reduceop was rerun!"

  def test_no_reduceop_rerun_alt(self):
    a = Tensor.randn(16, 16, 16)
    with CLCache() as cache:
      c = a.sum(2).permute(1,0)
      d = a.sum(2)
      c.realize()
      d.realize()
      cache_len = cache.count
    np.testing.assert_allclose(c.numpy(), d.numpy().transpose(1,0), rtol=1e-3, atol=1e-5)
    assert cache_len == 1, "reduceop was rerun!"

  def test_expand_reduce_is_folded_on_same_axis(self):
    with Context(FUSE_CONV_BW=1):
      for axis in [0, 1]:
        for n in [4, 8, 16]:
          b = torch.ones(n, n).sum(axis).reshape(n, 1).expand(n, n).sum(axis)
          with CLCache(allowed=2):
            a = Tensor.ones(n, n).contiguous().sum(axis).reshape(n, 1).expand(n, n).sum(axis)
            a.realize()
          np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-3, atol=1e-5)

  def test_expand_reduce_is_folded_on_different_axes(self):
    with Context(FUSE_CONV_BW=1):
      axis1, axis2 = 0, 1
      for n in [4, 8, 16]:
        b = torch.ones(n, n).sum(axis1).reshape(n, 1).expand(n, n).sum(axis2)
        with CLCache(allowed=2):
          a = Tensor.ones(n, n).contiguous().sum(axis1).reshape(n, 1).expand(n, n).sum(axis2)
          a.realize()
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-3, atol=1e-5)

if __name__ == '__main__':
  unittest.main()
