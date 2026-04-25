"""
``Parser.parse_vision_outputs`` contract tests (LOW-LEVEL §4.1 P1).

Maps: R1.
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.tests.modeld_parse_fixtures import minimal_vision_parse_outputs


def test_parse_vision_outputs_populates_expected_keys():
  outs = minimal_vision_parse_outputs(batch=1)
  Parser().parse_vision_outputs(outs)

  assert "pose" in outs and "pose_stds" in outs
  assert outs["pose"].shape == (1, ModelConstants.POSE_WIDTH)
  assert outs["wide_from_device_euler"].shape == (1, ModelConstants.WIDE_FROM_DEVICE_WIDTH)
  assert outs["desire_pred"].shape == (1, ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_PRED_WIDTH)
  assert outs["meta"].shape == (1, 55)


def test_parse_vision_outputs_desire_pred_rows_sum_to_one():
  outs = minimal_vision_parse_outputs(batch=1)
  outs["desire_pred"][0, :] = np.arange(ModelConstants.DESIRE_PRED_LEN * ModelConstants.DESIRE_PRED_WIDTH, dtype=np.float32)
  Parser().parse_vision_outputs(outs)

  dp = outs["desire_pred"]
  assert dp.shape == (1, ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_PRED_WIDTH)
  np.testing.assert_allclose(
    dp.sum(axis=-1),
    np.ones((1, ModelConstants.DESIRE_PRED_LEN), dtype=np.float32),
    atol=1e-5,
  )


def test_parse_vision_outputs_meta_is_sigmoid_bounded():
  outs = minimal_vision_parse_outputs(batch=1)
  rng = np.random.default_rng(1)
  outs["meta"][0, :] = rng.standard_normal(outs["meta"].shape[1], dtype=np.float32)
  Parser().parse_vision_outputs(outs)

  m = outs["meta"]
  assert m.shape == (1, 55)
  assert np.all(m > 0.0) and np.all(m < 1.0)


def test_parse_vision_outputs_pose_mdn_splits_mu_and_positive_stds():
  outs = minimal_vision_parse_outputs(batch=1)
  pw = ModelConstants.POSE_WIDTH
  raw = np.zeros(pw * 2, dtype=np.float32)
  raw[:pw] = np.array([0.5, -0.25, 0.0, 0.1, 0.2, 0.3], dtype=np.float32)
  raw[pw:] = np.log(np.array([0.5, 0.25, 0.125, 0.2, 0.15, 0.1], dtype=np.float32))
  outs["pose"][0, :] = raw
  Parser().parse_vision_outputs(outs)

  np.testing.assert_allclose(outs["pose"][0], raw[:pw], rtol=0, atol=1e-6)
  assert outs["pose_stds"].shape == (1, pw)
  assert np.all(outs["pose_stds"] > 0.0)
