from tinygrad import Tensor
from test.external.mlperf_retinanet.focal_loss import sigmoid_focal_loss as ref_sigmoid_focal_loss
from test.external.mlperf_unet3d.dice import DiceCELoss
from examples.mlperf.losses import dice_ce_loss, sigmoid_focal_loss, l1_loss

import numpy as np
import torch
import unittest

class ExternalTestLosses(unittest.TestCase):
  def setUp(self):
    np.random.seed(42)

  def _assert_loss(self, pred, tgt, tinygrad_metrics, ref_metrics, rtol=1e-07, atol=0, **kwargs):
    tinygrad_metrics_res = tinygrad_metrics(Tensor(pred), Tensor(tgt), **kwargs)
    ref_metrics_res = ref_metrics(torch.from_numpy(pred), torch.from_numpy(tgt), **kwargs)
    np.testing.assert_allclose(tinygrad_metrics_res.numpy(), ref_metrics_res.numpy(), rtol=rtol, atol=atol)

  def test_dice_ce_loss(self):
    pred, label = np.random.rand(1, 3, 128, 128, 128).astype(np.float32), np.ones((1, 1, 128, 128, 128)).astype(np.uint8)
    tinygrad_metrics_res, ref_metrics_res = dice_ce_loss, DiceCELoss(True, True, "NCDHW", False)
    self._assert_loss(pred, label, tinygrad_metrics_res, ref_metrics_res, atol=1e-4)

  def test_sigmoid_focal_loss(self):
    def _apply_logit(p): return np.log(p / (1 - p))
    pred, tgt = _apply_logit(np.random.rand(5,2).astype(np.float32)), np.random.randint(0, 2, (5, 2)).astype(np.float32)
    for reduction in ["mean", "sum", "none"]:
      for alpha, gamma in zip([-1, 0.58], [0, 2]):
        self._assert_loss(pred, tgt, sigmoid_focal_loss, ref_sigmoid_focal_loss, rtol=1e-4, alpha=alpha, gamma=gamma, reduction=reduction)

  def test_l1_loss(self):
    N, C, H, W = 3, 4, 5, 6
    shapes = ((N, C), (N, C, H), (N, C, H, W))
    for reduction in ["mean", "sum", "none"]:
      for shape in shapes:
        pred, tgt = np.random.randint(shape).astype(np.float32), np.random.randint(shape)
        self._assert_loss(pred, tgt, l1_loss, torch.nn.functional.l1_loss, reduction=reduction)

if __name__ == '__main__':
  unittest.main()