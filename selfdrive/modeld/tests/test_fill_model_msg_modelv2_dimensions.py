"""
``ModelConstants``-driven list sizes on ``modelV2`` after ``fill_model_msg``.

New tests only; upstream ``test_fill_model_msg.py`` unchanged.

``docs/testing/testing-plan/TESTING-PLAN.md`` §3.1 / tracker Phase B.

Maps: R1.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg
from openpilot.selfdrive.modeld.tests.modeld_test_fixtures import DummyBuilder, minimal_net_output_data


def test_modelv2_collection_lengths_match_constants():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  state = PublishState()

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=11.0,
    delay=0.05,
    publish_state=state,
    vipc_frame_id=10,
    vipc_frame_id_extra=10,
    frame_id=10,
    frame_drop=0.0,
    timestamp_eof=999,
    model_execution_time=0.015,
    valid=True,
  )

  mv2 = ext.modelV2
  assert len(mv2.laneLines) == ModelConstants.NUM_LANE_LINES
  assert len(mv2.roadEdges) == ModelConstants.NUM_ROAD_EDGES
  assert len(mv2.leadsV3) == ModelConstants.LEAD_MHP_SELECTION
  assert len(mv2.laneLineProbs) == ModelConstants.NUM_LANE_LINES
  assert len(mv2.meta.desirePrediction) == ModelConstants.DESIRE_PRED_LEN * ModelConstants.DESIRE_PRED_WIDTH


def test_net_output_batch_dim_supported_for_synthetic_data():
  """``fill_model_msg`` reads batch index 0; tensors may still carry batch > 1 for contract tests."""
  outs = minimal_net_output_data(batch=3)
  assert outs["plan"].shape[0] == 3
  assert outs["desired_curvature"].shape[0] == 3


def test_plan_xyzt_time_axis_and_first_waypoint_match_net_output():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  state = PublishState()

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=3.0,
    delay=0.0,
    publish_state=state,
    vipc_frame_id=5,
    vipc_frame_id_extra=5,
    frame_id=5,
    frame_drop=0.0,
    timestamp_eof=123,
    model_execution_time=0.02,
    valid=True,
  )

  mv2 = ext.modelV2
  assert mv2.position.t == ModelConstants.T_IDXS
  pos0 = outs["plan"][0, 0, Plan.POSITION]
  assert mv2.position.x[0] == pytest.approx(float(pos0[0]))
  assert mv2.position.y[0] == pytest.approx(float(pos0[1]))
  assert mv2.position.z[0] == pytest.approx(float(pos0[2]))
  vel0 = outs["plan"][0, 0, Plan.VELOCITY]
  assert mv2.velocity.x[0] == pytest.approx(float(vel0[0]))
  assert mv2.velocity.t == ModelConstants.T_IDXS


def test_temporal_pose_splits_sim_pose_and_stds():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  outs["sim_pose"][0, :] = np.array([11.0, 12.0, 13.0, 0.4, 0.5, 0.6], dtype=np.float32)
  outs["sim_pose_stds"][0, :] = np.array([0.7, 0.8, 0.9, 0.11, 0.22, 0.33], dtype=np.float32)

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=4.0,
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

  tp = ext.modelV2.temporalPose
  half = ModelConstants.POSE_WIDTH // 2
  np.testing.assert_allclose(tp.trans, outs["sim_pose"][0, :half].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(tp.transStd, outs["sim_pose_stds"][0, :half].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(tp.rot, outs["sim_pose"][0, half:].tolist(), rtol=0, atol=0)
  np.testing.assert_allclose(tp.rotStd, outs["sim_pose_stds"][0, half:].tolist(), rtol=0, atol=0)


def test_leads_v3_prob_and_prob_time_contract():
  base = DummyBuilder()
  ext = DummyBuilder()
  outs = minimal_net_output_data(batch=1)
  outs["lead_prob"][0, :] = np.array([0.11, 0.22, 0.33], dtype=np.float32)

  fill_model_msg(
    base_msg=base,
    extended_msg=ext,
    net_output_data=outs,
    v_ego=6.0,
    delay=0.0,
    publish_state=PublishState(),
    vipc_frame_id=2,
    vipc_frame_id_extra=2,
    frame_id=2,
    frame_drop=0.0,
    timestamp_eof=1,
    model_execution_time=0.01,
    valid=True,
  )

  for i, lead in enumerate(ext.modelV2.leadsV3):
    assert lead.probTime == ModelConstants.LEAD_T_OFFSETS[i]
    np.testing.assert_allclose(lead.prob, outs["lead_prob"][0, i].tolist(), rtol=0, atol=0)
    assert lead.t == ModelConstants.LEAD_T_IDXS
