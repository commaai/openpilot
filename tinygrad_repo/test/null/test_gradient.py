from typing import Callable
import unittest, math
import torch
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.gradient import compute_gradient

class TestGradient(unittest.TestCase):
  def _cmp_nan_okay(self, x, y):
    if math.isnan(x) and math.isnan(y): return
    self.assertAlmostEqual(x, y, places=5)

  def _test_one_input_function(self, f:Callable, jf:Callable|None=None):
    if jf is None: jf = f
    x = UOp.variable('x', -math.inf, math.inf, dtype=dtypes.float)
    gx = compute_gradient(f(x), UOp.const(dtypes.float, 1.0), set([x]))[x]

    for val in [-5., -2.0, 0.0, 2.0, 5.]:
      tg_out = gx.substitute({x: x.const_like(val)}).ssimplify()
      tx = torch.tensor([val], dtype=torch.float, requires_grad=True)
      torch_out = torch.autograd.grad(jf(tx), tx)[0].item()
      self._cmp_nan_okay(tg_out, torch_out)

  def _test_two_input_function(self, f:Callable, jf:Callable|None=None):
    if jf is None: jf = f
    x = UOp.variable('x', -math.inf, math.inf, dtype=dtypes.float)
    y = UOp.variable('y', -math.inf, math.inf, dtype=dtypes.float)
    grads = compute_gradient(f(x, y), UOp.const(dtypes.float, 1.0), set([x, y]))
    gx, gy = grads[x], grads[y]

    for valx in [-5., -2.0, 0.0, 2.0, 5.]:
      for valy in [-5., -2.0, 0.0, 2.0, 5.]:
        # Substitute the values into the gradient expressions
        substitutions = {x: x.const_like(valx), y: y.const_like(valy)}
        tg_out_x = gx.substitute(substitutions).ssimplify()
        tg_out_y = gy.substitute(substitutions).ssimplify()

        tx = torch.tensor([valx], dtype=torch.float, requires_grad=True)
        ty = torch.tensor([valy], dtype=torch.float, requires_grad=True)
        torch_grad = torch.autograd.grad(jf(tx, ty), [tx, ty])
        torch_out_x, torch_out_y = [x.item() for x in torch_grad]

        self._cmp_nan_okay(tg_out_x, torch_out_x)
        self._cmp_nan_okay(tg_out_y, torch_out_y)

  # unary ops unit
  def test_recip(self): self._test_one_input_function(lambda x: 1.0/x)
  def test_sin(self): self._test_one_input_function(lambda x: x.sin())
  def test_sqrt(self): self._test_one_input_function(lambda x: x.sqrt())
  def test_log2(self): self._test_one_input_function(lambda x: x.log2())
  def test_exp2(self): self._test_one_input_function(lambda x: x.exp2())

  # binary ops unit
  def test_add(self): self._test_two_input_function(lambda x,y: x+y)
  def test_mul(self): self._test_two_input_function(lambda x,y: x*y)

  # chain rule
  def test_chain(self): self._test_one_input_function(lambda x: x.sin().sqrt())
  def test_chain_binop(self): self._test_two_input_function(lambda x,y: (x*y)+x*y)
  def test_big_add_sin(self): self._test_two_input_function(lambda x,y: x.sin()+3.0/y)
  def test_big_chain(self): self._test_two_input_function(lambda x,y: (1.0/x*y)+x*y)
  def test_where(self): self._test_two_input_function(lambda x,y: (x<y).where(x,y), lambda x,y: torch.where(x<y,x,y))

class TestRealizeMeansRealize(unittest.TestCase):
  def test_randn_realizes(self):
    x = Tensor.randn(2, 3, 64, 64, requires_grad=True).realize()
    assert x.uop is not x.uop.base
    assert x.uop.is_realized

  def test_uniform_realizes(self):
    x = Tensor.uniform(16, 3, 3, 3, requires_grad=True).realize()
    print(x.uop)
    assert x.uop is not x.uop.base
    assert x.uop.is_realized

  def test_uniform_gradient(self):
    x = Tensor.uniform(16, 3, 3, 3, requires_grad=True).realize()
    y = x * 2
    y.sum().gradient(x)[0].realize()

if __name__ == '__main__':
  unittest.main()
