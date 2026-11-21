from tinygrad import Tensor
from test.external.mlperf_unet3d.dice import DiceScore
from examples.mlperf.metrics import dice_score, log_perplexity

import numpy as np
import torch
import unittest, math

class ExternalTestMetrics(unittest.TestCase):
  def _test_metrics(self, tinygrad_metrics, orig_metrics, pred, label, atol=1e-8, rtol=1e-7):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(label)).squeeze().numpy()
    orig_metrics_res = orig_metrics(torch.from_numpy(pred), torch.from_numpy(label)).numpy()
    np.testing.assert_allclose(tinygrad_metrics_res, orig_metrics_res, atol=atol, rtol=rtol)

  def test_dice(self):
    pred, label = np.random.rand(1, 3, 128, 128, 128).astype(np.float32), np.ones((1, 1, 128, 128, 128)).astype(np.uint8)
    self._test_metrics(dice_score, DiceScore(), pred, label)

  def test_log_perplexity(self):
    # equally likely
    np.testing.assert_allclose(log_perplexity(Tensor([[[1.0, 1, 1, 1]]]), Tensor([[2]])).numpy(), math.log(4))
    np.testing.assert_allclose(log_perplexity(Tensor([[[1.0]*256]*32]), Tensor([[2]*32])).numpy(), math.log(256), rtol=1e-6)
    # pretty correct and incorrect
    np.testing.assert_allclose(log_perplexity(Tensor([[[10000., 0, 0, 0]]]), Tensor([[0]])).numpy(), 0)
    np.testing.assert_allclose(log_perplexity(Tensor([[[0.0, 10000, 10000, 10000]]]), Tensor([[0]])).numpy(), 10000, rtol=1e-3)
    # higher logit -> lower loss
    x = Tensor([[[4.0, 3, 2, 1]]])
    for i in range(x.numel()-1): self.assertLess(log_perplexity(x, Tensor([[i]])).item(), log_perplexity(x, Tensor([[i+1]])).item())
    # torch eval examples
    np.testing.assert_allclose(
      log_perplexity(Tensor([[[0.3659, 0.7025, 0.3104], [0.0097, 0.6577, 0.1947]]]), Tensor([[2, 1]])).exp().numpy(),
      2.7593, rtol=1e-5)
    np.testing.assert_allclose(
      log_perplexity(Tensor([[[0.3, 0.7, 0.3, 0.1], [0.5, 0.4, 0.1, 0.4],[0.1, 0.1, 0.2, 0.5]],
                             [[0.1, 0.6, 0.1, 0.5], [0.3, 0.7, 0.3, 0.4], [0.3, 0.7, 0.3, 0.4]]]), Tensor([[2, 1, 3],  [1, 0, 1]])).exp().numpy(),
      3.6216, rtol=1e-5)
    np.testing.assert_allclose(
      log_perplexity(Tensor([[[0.3659, 0.7025, 0.3104], [0.0097, 0.6577, 0.1947]]]), Tensor([[2, 1]]), ignore_index=1).exp().numpy(),
      3.5372, rtol=1e-4)

if __name__ == '__main__':
  unittest.main()