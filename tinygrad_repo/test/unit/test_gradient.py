from typing import Callable
import unittest, math
import jax
import jax.numpy as jnp
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Ops
from tinygrad.gradient import compute_gradient

class TestGradient(unittest.TestCase):
  def _cmp_nan_okay(self, x, y):
    if math.isnan(x) and math.isnan(y): return
    self.assertAlmostEqual(x, y, places=5)

  def _test_one_input_function(self, f:Callable, jf:Callable|None=None):
    x = UOp.variable('x', -math.inf, math.inf, dtype=dtypes.float)
    gx = compute_gradient(f(x), UOp.const(dtypes.float, 1.0), [x])[x]
    gf = jax.grad(f if jf is None else jf)

    for val in [-5., -2.0, 0.0, 2.0, 5.]:
      tg_out, jax_out = gx.substitute({x: x.const_like(val)}).ssimplify(), gf(val).item()
      self._cmp_nan_okay(tg_out, jax_out)

  def _test_two_input_function(self, f:Callable, jf:Callable|None=None):
    x = UOp.variable('x', -math.inf, math.inf, dtype=dtypes.float)
    y = UOp.variable('y', -math.inf, math.inf, dtype=dtypes.float)
    grads = compute_gradient(f(x, y), UOp.const(dtypes.float, 1.0), [x, y])
    gx, gy = grads[x], grads[y]
    gf = jax.grad(f if jf is None else jf, argnums=(0, 1))

    for valx in [-5., -2.0, 0.0, 2.0, 5.]:
      for valy in [-5., -2.0, 0.0, 2.0, 5.]:
        # Substitute the values into the gradient expressions
        substitutions = {x: x.const_like(valx), y: y.const_like(valy)}
        tg_out_x = gx.substitute(substitutions).ssimplify()
        tg_out_y = gy.substitute(substitutions).ssimplify()
        jax_out_x, jax_out_y = [x.item() for x in gf(valx, valy)]

        self._cmp_nan_okay(tg_out_x, jax_out_x)
        self._cmp_nan_okay(tg_out_y, jax_out_y)

  # unary ops unit
  def test_recip(self): self._test_one_input_function(lambda x: 1.0/x)
  def test_sin(self): self._test_one_input_function(lambda x: x.sin(), lambda x: jnp.sin(x))
  def test_sqrt(self): self._test_one_input_function(lambda x: x.sqrt(), lambda x: jnp.sqrt(x))
  def test_log2(self): self._test_one_input_function(lambda x: x.log2(), lambda x: jnp.log2(x))
  def test_exp2(self): self._test_one_input_function(lambda x: x.exp2(), lambda x: jnp.exp2(x))

  # binary ops unit
  def test_add(self): self._test_two_input_function(lambda x,y: x+y)
  def test_mul(self): self._test_two_input_function(lambda x,y: x*y)

  # chain rule
  def test_chain(self): self._test_one_input_function(lambda x: x.sin().sqrt(), lambda x: jnp.sqrt(jnp.sin(x)))
  def test_chain_binop(self): self._test_two_input_function(lambda x,y: (x*y)+x*y)
  def test_big_add_sin(self): self._test_two_input_function(lambda x,y: x.sin()+3.0/y, lambda x,y: jnp.sin(x)+3.0/y)
  def test_big_chain(self): self._test_two_input_function(lambda x,y: (1.0/x*y)+x*y)
  def test_where(self): self._test_two_input_function(lambda x,y: (x<y).where(x,y), lambda x,y: jnp.where(x<y, x, y))

class TestTensorGradient(unittest.TestCase):
  def test_example(self):
    x = Tensor.eye(3)
    y = Tensor([[2.0,0,-2.0]])
    z = y.matmul(x).sum()
    dx, dy = z.gradient(x, y)
    self.assertListEqual(dx.tolist(), [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
    self.assertListEqual(dy.tolist(), [[1.0, 1.0, 1.0]])

  def test_raises(self):
    x = Tensor([1.0, 2.0, 3.0])
    w = Tensor.randn((3,))
    with self.assertRaises(RuntimeError): x.sum().gradient(w)

  def test_with_custom_gradient(self):
    x = Tensor([1.0, 2.0, 3.0])
    z = (x * x).sum()
    dx = z.gradient(x, gradient=Tensor([3.0]))[0]
    self.assertListEqual(dx.tolist(), [6.0, 12.0, 18.0])

  def test_broadcast_gradient(self):
    x = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[10.0, 20.0, 30.0, 40.0]])
    z = (x + y).sum()
    dx, dy = z.gradient(x, y)
    self.assertListEqual(dx.tolist(), [[4.0], [4.0], [4.0]])
    self.assertListEqual(dy.tolist(), [[3.0, 3.0, 3.0, 3.0]])

  def test_non_scalar_output(self):
    x = Tensor([1.0, 2.0, 3.0])
    z = x * x
    with self.assertRaises(AssertionError): z.gradient(x)
    dz = Tensor([1.0, 1.0, 1.0])
    dx = z.gradient(x, gradient=dz)[0]
    self.assertListEqual(dx.tolist(), [2.0, 4.0, 6.0])

class TestRealizeMeansRealize(unittest.TestCase):
  def test_randn_realizes(self):
    x = Tensor.randn(2, 3, 64, 64, requires_grad=True).realize()
    self.assertEqual(x.lazydata.op, Ops.VIEW)

  @unittest.expectedFailure
  def test_uniform_realizes(self):
    x = Tensor.uniform(16, 3, 3, 3, requires_grad=True).realize()
    print(x.lazydata)
    self.assertEqual(x.lazydata.op, Ops.VIEW)

if __name__ == '__main__':
  unittest.main()
