from tinygrad import Tensor
from test.external.mlperf_unet3d.dice import DiceScore
from examples.mlperf.metrics import dice_score

import numpy as np
import torch
import unittest

class ExternalTestMetrics(unittest.TestCase):
  def _test_metrics(self, tinygrad_metrics, orig_metrics, pred, label, atol=1e-8, rtol=1e-7):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(label)).squeeze().numpy()
    orig_metrics_res = orig_metrics(torch.from_numpy(pred), torch.from_numpy(label)).numpy()
    np.testing.assert_allclose(tinygrad_metrics_res, orig_metrics_res, atol=atol, rtol=rtol)

  def test_dice(self):
    pred, label = np.random.rand(1, 3, 128, 128, 128).astype(np.float32), np.ones((1, 1, 128, 128, 128)).astype(np.uint8)
    self._test_metrics(dice_score, DiceScore(), pred, label)

if __name__ == '__main__':
  unittest.main()