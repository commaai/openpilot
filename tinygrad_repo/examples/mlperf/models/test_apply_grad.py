import unittest
from tinygrad import Tensor, TinyJit
from tinygrad.nn.state import get_parameters
from examples.mlperf.models.flat_llama import apply_grad

class FlatModel:
  def __init__(self, n_layers:int, dim:int, hidden:int):
    self.n_layers = n_layers
    self.w1 = Tensor.uniform(n_layers, dim, hidden, low=-0.1, high=0.1)
    self.w2 = Tensor.uniform(n_layers, hidden, dim, low=-0.1, high=0.1)
    self.scale = Tensor.uniform(dim, low=0.9, high=1.1)
    self.bias = Tensor.zeros(dim).contiguous()

  def __call__(self, x:Tensor) -> Tensor:
    h = x
    for i in range(self.n_layers):
      h = (h @ self.w1[i]).relu() @ self.w2[i] + h
    return (h * self.scale + self.bias).sum()

class TestApplyGradE2E(unittest.TestCase):
  def _run_with_apply_grad(self, model, xs):
    grads = {p: Tensor.zeros(p.shape, dtype=p.dtype).contiguous().realize() for p in get_parameters(model)}
    for x in xs:
      loss = model(x)
      for p, g in zip(grads, loss.gradient(*grads)):
        apply_grad(grads[p], g.uop)
      Tensor.realize(loss, *grads.values())
    return [grads[p] for p in get_parameters(model)]

  def _run_reference(self, model, xs):
    for x in xs: model(x).backward()
    return [p.grad for p in get_parameters(model)]

  def _assert_close(self, got, expected, atol, rtol):
    for g, e in zip(got, expected):
      self.assertTrue(g.allclose(e, atol=atol, rtol=rtol).item(), f"grad mismatch (max abs diff {(g - e).abs().max().item()})")

  def _assert_match(self, model, xs, atol, rtol):
    self._assert_close(self._run_with_apply_grad(model, xs), self._run_reference(model, xs), atol, rtol)

  def test_e2e_single_step(self):
    model = FlatModel(n_layers=3, dim=8, hidden=16)
    Tensor.realize(*get_parameters(model))
    self._assert_match(model, [Tensor.randn(2, 8).realize()], atol=1e-4, rtol=1e-4)

  def test_e2e_multi_step_accumulation(self):
    model = FlatModel(n_layers=4, dim=8, hidden=16)
    Tensor.realize(*get_parameters(model))
    self._assert_match(model, [Tensor.randn(2, 8).realize() for _ in range(3)], atol=1e-4, rtol=1e-4)

  def test_e2e_jit(self):
    model = FlatModel(n_layers=3, dim=8, hidden=16)
    Tensor.realize(*get_parameters(model))
    grads = {p: Tensor.zeros(p.shape, dtype=p.dtype).contiguous().realize() for p in get_parameters(model)}

    @TinyJit
    def fwd_bwd(x:Tensor):
      loss = model(x)
      for p, g in zip(grads, loss.gradient(*grads)): apply_grad(grads[p], g.uop)
      Tensor.realize(loss, *grads.values())

    xs = [Tensor.randn(2, 8).realize() for _ in range(3)]
    for x in xs: fwd_bwd(x)
    self._assert_close([grads[p] for p in get_parameters(model)], self._run_reference(model, xs), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
  unittest.main()
