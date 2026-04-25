"""
``drivingModelData`` fields mirrored from ``fill_model_msg`` arguments.

LOW-LEVEL §7.1 Phase B.

Maps: R1.
"""

from __future__ import annotations

import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg
from openpilot.selfdrive.modeld.tests.modeld_test_fixtures import DummyBuilder, minimal_net_output_data


def test_driving_model_data_frame_ids_follow_vipc():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  state = PublishState()

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=8.0,
    delay=0.0,
    publish_state=state,
    vipc_frame_id=700,
    vipc_frame_id_extra=701,
    frame_id=700,
    frame_drop=0.0,
    timestamp_eof=123,
    model_execution_time=0.03,
    valid=True,
  )

  dm = base.drivingModelData
  assert dm.frameId == 700
  assert dm.frameIdExtra == 701
  assert dm.modelExecutionTime == pytest.approx(0.03, abs=1e-6)
  assert dm.action.desiredCurvature == pytest.approx(float(outs["desired_curvature"][0, 0]), abs=1e-6)


def test_driving_model_data_valid_flag_mirrors_fill_model_msg():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=0.0,
    delay=0.0,
    publish_state=PublishState(),
    vipc_frame_id=1,
    vipc_frame_id_extra=1,
    frame_id=1,
    frame_drop=0.0,
    timestamp_eof=1,
    model_execution_time=0.0,
    valid=False,
  )

  assert base.valid is False
  assert ext.valid is False


def test_lane_line_meta_matches_lane_lines_and_probs():
  """``fill_lane_line_meta`` uses lines 1/2 at lateral ``y[0]`` (tensor column 0) and probs 1/2."""
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  outs["lane_lines"][0, 1, 0, 0] = 1.25
  outs["lane_lines"][0, 2, 0, 0] = -2.5

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=1.0,
    delay=0.0,
    publish_state=PublishState(),
    vipc_frame_id=3,
    vipc_frame_id_extra=3,
    frame_id=3,
    frame_drop=0.0,
    timestamp_eof=2,
    model_execution_time=0.001,
    valid=True,
  )

  probs = outs["lane_lines_prob"][0, 1::2].tolist()
  meta = base.drivingModelData.laneLineMeta
  assert meta.leftY == pytest.approx(1.25)
  assert meta.rightY == pytest.approx(-2.5)
  assert meta.leftProb == pytest.approx(float(probs[1]))
  assert meta.rightProb == pytest.approx(float(probs[2]))


def test_path_polynomial_degree_matches_model_constants():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=2.0,
    delay=0.0,
    publish_state=PublishState(),
    vipc_frame_id=1,
    vipc_frame_id_extra=1,
    frame_id=1,
    frame_drop=0.0,
    timestamp_eof=1,
    model_execution_time=0.01,
    valid=True,
  )

  path = base.drivingModelData.path
  assert len(path.xCoefficients) == ModelConstants.POLY_PATH_DEGREE + 1
  assert len(path.yCoefficients) == ModelConstants.POLY_PATH_DEGREE + 1
  assert len(path.zCoefficients) == ModelConstants.POLY_PATH_DEGREE + 1
