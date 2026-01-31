import unittest
import numpy as np
from tinygrad import Tensor, UOp, nn
from tinygrad.uop.ops import AxisType, Ops

class TestOuterworldReduce(unittest.TestCase):
  def test_reduce(self):
    x = Tensor.ones(5, 5).contiguous()
    a = UOp.range(5, -1, AxisType.REDUCE)
    out = x[a]
    # TODO: syntax for this
    t = Tensor(UOp(Ops.REDUCE, dtype=out.uop.dtype, src=(out.uop, a), arg=Ops.ADD))
    self.assertListEqual(t.tolist(), [5.,5.,5.,5.,5.])

# TODO: delete test_outerworld_range?
class TestOuterRange(unittest.TestCase):
  def test_simple_range(self):
    a = Tensor.ones(10).contiguous()
    acc = Tensor.zeros().contiguous()
    Tensor.realize(a, acc)

    # this is fold
    i = UOp.range(10, -100, AxisType.OUTER)
    acc_i = acc.uop.after(i)
    vi = UOp.variable("i", i.vmin, i.vmax).bind(i)
    out = Tensor(acc.uop.after(acc_i.store(acc_i + a[vi].uop).end(i)))
    out.realize()
    assert out.item() == 10.0

  def test_inner_range(self):
    a = Tensor.ones(10, 10).contiguous()
    acc = Tensor.zeros(10).contiguous()
    Tensor.realize(a, acc)

    # this is fold
    i = UOp.range(10, -100, AxisType.OUTER)
    acc_i = acc.uop.after(i)
    vi = UOp.variable("i", i.vmin, i.vmax).bind(i)
    out = Tensor(acc.uop.after(acc_i.store(acc_i + a[:, vi].uop).end(i)))
    out.realize()
    assert all(x == 10.0 for x in out.tolist())

  def test_range_matmul(self):
    vec = Tensor.randn(1, 10).realize()
    mats = Tensor.randn(3, 10, 10).realize()

    # 3 matmuls in "scan"
    ref = ((vec @ mats[0]) @ mats[1]) @ mats[2]
    ref.realize()

    # 3 matmuls with outer world range
    i = UOp.range(3, -100, AxisType.OUTER)
    vec_i = Tensor(vec.uop.after(i))
    comp = vec_i.contiguous() @ mats[i]
    store = vec_i.uop.store(comp.uop).end(i)
    out = Tensor(vec.uop.after(store))
    out.realize()

    # TODO: testing allclose
    assert Tensor.allclose(ref, out, atol=1e-6), f"{ref.numpy()=}, {out.numpy()=}"

class TestOuterScan(unittest.TestCase):
  def _test_scan(self):
    vec = Tensor.randn(1, 10).realize()
    mats = Tensor.randn(3, 10, 10).realize()

    # 3 matmuls in "scan"
    vec1 = vec @ mats[0]
    vec2 = vec1 @ mats[1]
    vec3 = vec2 @ mats[2]
    ref = Tensor.stack(vec1, vec2, vec3)
    ref.realize()
    return vec, mats, ref

  def test_uop_scan_matmul(self):
    vec, mats, ref = self._test_scan()

    # 3 matmuls with SCAN
    i = UOp.range(3, -100, AxisType.OUTER)
    out = Tensor.empty(3, 1, 10)
    phi = Tensor(i.eq(0).where(vec.uop, out[(i-1).maximum(0)].uop))
    comp = phi @ mats[i]
    store = out[i].uop.store(comp.uop).end(i)
    out = Tensor(out.uop.after(store))
    out.realize()

    # TODO: testing allclose
    assert Tensor.allclose(ref, out, atol=1e-6), f"{ref.numpy()=}, {out.numpy()=}"

class TestOuterworld(unittest.TestCase):
  def test_range_plus_1(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t+1==cpy).all().item())

  def test_range_plus_1_transpose(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(10, 1).expand(10, a).contiguous().realize()

    self.assertTrue(((t+1).T==cpy).all().item())

  def test_flip_range(self):
    t = Tensor.rand(10, 10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[9-a]
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t.flip(0)==cpy).all().item())

  def test_vmap(self):
    def f(x): return x.sum(axis=0)*2

    x = Tensor.ones(3, 10, 2).contiguous()

    # vmap across axis 0
    a = UOp.range(3, -1)
    out = f(x[a])
    out = out.reshape(1, 2).expand(a, 2).contiguous()

    # 3x2 grid of 20
    out.realize()
    self.assertTrue((out==20).all().item())

  def test_fancy_vmap(self):
    def f(x,y): return x+y

    x = Tensor.arange(9).reshape(3,3).contiguous()
    y = Tensor.arange(9).reshape(3,3).contiguous()

    a = UOp.range(3, -1)
    out = f(x[:,a], y[a,:])
    # TODO: this should support flatten
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    self.assertListEqual([[0,4,8],[4,8,12],[8,12,16]], out.tolist())

class TestVmap(unittest.TestCase):
  def test_vmap_inner(self, axis_type=AxisType.LOOP, fuse=False, grad=False):
    x = Tensor.ones(1, 10).contiguous().requires_grad_()
    mats = Tensor.ones(3, 10, 10).contiguous().requires_grad_()

    ref = x @ mats
    if fuse: ref = ref * 2

    # vmap across axis 0
    a = UOp.range(3, -1, axis_type)
    out = x @ mats[a]
    out = out.reshape(1, 10).pad(((a,(3-a)-1), None))
    out = Tensor(out.uop.reduce(a, arg=Ops.ADD))
    if fuse: out = out * 2
    if grad:
      out.mean().backward()
      np.testing.assert_allclose(mats.grad.numpy(), (2./30) if fuse else (1./30))
    out.realize()

    # TODO: testing allclose
    assert Tensor.allclose(ref, out, atol=1e-6), f"{ref.numpy()=}, {out.numpy()=}"
  def test_vmap_inner_fuse(self): self.test_vmap_inner(fuse=True)
  def test_vmap_outer(self): self.test_vmap_inner(AxisType.OUTER)
  def test_vmap_outer_fuse(self): self.test_vmap_inner(AxisType.OUTER, fuse=True)

  def test_vmap_inner_grad(self): self.test_vmap_inner(grad=True)
  def test_vmap_inner_fuse_grad(self): self.test_vmap_inner(fuse=True, grad=True)
  def test_vmap_outer_grad(self): self.test_vmap_inner(AxisType.OUTER, grad=True)

  def test_vmap_convs(self):
    layers = [
      nn.Conv2d(1, 8, 3), Tensor.relu,
      nn.Conv2d(8, 8, 3), Tensor.relu]
    img = Tensor.randn(4, 1, 16, 16).realize(*nn.state.get_parameters(layers))
    a = UOp.range(4, -1, AxisType.OUTER)
    out = img[a:a+1].sequential(layers)
    out = out.pad(((a,(4-a)-1), None, None, None))
    out = Tensor(out.uop.reduce(a, arg=Ops.ADD))
    out.realize()
    np.testing.assert_allclose(out.numpy(), img.sequential(layers).numpy(), atol=1e-6)

  def test_vmap_gemm(self):
    layers = [
      nn.Linear(16, 16, bias=False), Tensor.relu,
      nn.Linear(16, 16, bias=False), Tensor.relu]
    img = Tensor.randn(4, 16).realize(*nn.state.get_parameters(layers))
    a = UOp.range(4, -1, AxisType.OUTER)
    out = img[a:a+1].sequential(layers)
    out = out.pad(((a,(4-a)-1), None))
    out = Tensor(out.uop.reduce(a, arg=Ops.ADD))
    out.realize()
    np.testing.assert_allclose(out.numpy(), img.sequential(layers).numpy(), atol=1e-6)

  @unittest.skip("this is broken, we need to lower the outer reduce in the outer graph")
  def test_vmap_gemm_grad(self):
    layers = [
      nn.Linear(16, 16, bias=False), Tensor.relu,
      nn.Linear(16, 16, bias=False), Tensor.relu]
    layer_tensors = nn.state.get_parameters(layers)
    img = Tensor.randn(4, 16).realize(*layer_tensors)
    for l in layer_tensors: l.requires_grad_()
    a = UOp.range(4, -1, AxisType.OUTER)
    out = img[a:a+1].sequential(layers)
    out = out.pad(((a,(4-a)-1), None))
    out = Tensor(out.uop.reduce(a, arg=Ops.ADD))
    out.mean().backward()
    grads = [l.grad for l in layer_tensors]
    out.realize(*grads)
    out_grads = [x.numpy() for x in grads]

    # compute reference grads
    for l in layer_tensors: l.grad = None
    img.sequential(layers).mean().backward()
    grads = [l.grad for l in layer_tensors]
    out.realize(*grads)
    ref_grads = [x.numpy() for x in grads]

    # compare
    for o,r in zip(out_grads, ref_grads): np.testing.assert_allclose(o, r, atol=1e-6)

if __name__ == '__main__':
  unittest.main()