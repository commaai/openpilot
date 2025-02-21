from tinygrad import Tensor
from test.external.mlperf_unet3d.dice import DiceCELoss
from examples.mlperf.losses import dice_ce_loss

import numpy as np
import torch
import unittest

class ExternalTestLosses(unittest.TestCase):
  def _test_losses(self, tinygrad_metrics, orig_metrics, pred, label):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(label)).numpy()
    orig_metrics_res = orig_metrics(torch.from_numpy(pred), torch.from_numpy(label)).numpy()
    np.testing.assert_allclose(tinygrad_metrics_res, orig_metrics_res, atol=1e-4)

  def test_dice_ce(self):
    pred, label = np.random.rand(1, 3, 128, 128, 128).astype(np.float32), np.ones((1, 1, 128, 128, 128)).astype(np.uint8)
    self._test_losses(dice_ce_loss, DiceCELoss(True, True, "NCDHW", False), pred, label)

if __name__ == '__main__':
  unittest.main()