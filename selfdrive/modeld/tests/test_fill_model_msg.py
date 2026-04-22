from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.modeld.constants import Meta, ModelConstants, Plan
from openpilot.selfdrive.modeld.fill_model_msg import (
  ConfidenceClass,
  PublishState,
  fill_lane_line_meta,
  fill_model_msg,
  fill_pose_msg,
  fill_xyvat,
  fill_xyzt,
  fill_xyz_poly,
)


class DummyBuilder:
  def __init__(self):
    super().__setattr__("_attrs", {})

  def __getattr__(self, name):
    if name.startswith("_"):
      raise AttributeError(name)
    if name not in self._attrs:
      self._attrs[name] = DummyBuilder()
    return self._attrs[name]

  def __setattr__(self, name, value):
    self._attrs[name] = value

  def init(self, name, count=None):
    # capnp supports both list init (field, count) and struct init (field).
    if count is None:
      value = DummyBuilder()
      self._attrs[name] = value
      return value

    values = [DummyBuilder() for _ in range(count)]
    self._attrs[name] = values
    return values


def _build_model_output_data() -> dict[str, np.ndarray]:
  idx_n = ModelConstants.IDX_N

  plan = np.zeros((1, idx_n, ModelConstants.PLAN_WIDTH), dtype=np.float32)
  plan_stds = np.full_like(plan, 0.1)
  for i, x in enumerate(ModelConstants.X_IDXS):
    plan[0, i, Plan.POSITION] = np.array([x, 0.1 * i, 0.0], dtype=np.float32)
    plan[0, i, Plan.VELOCITY] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    plan[0, i, Plan.ACCELERATION] = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    plan[0, i, Plan.T_FROM_CURRENT_EULER] = np.array([0.0, 0.0, 0.01], dtype=np.float32)
    plan[0, i, Plan.ORIENTATION_RATE] = np.array([0.0, 0.0, 0.001], dtype=np.float32)

  data = {
    "desired_curvature": np.array([[0.02]], dtype=np.float32),
    "plan": plan,
    "plan_stds": plan_stds,
    "sim_pose": np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32),
    "sim_pose_stds": np.full((1, ModelConstants.POSE_WIDTH), 0.05, dtype=np.float32),
    "lane_lines": np.zeros((1, ModelConstants.NUM_LANE_LINES, idx_n, ModelConstants.LANE_LINES_WIDTH), dtype=np.float32),
    "lane_lines_stds": np.full((1, ModelConstants.NUM_LANE_LINES, idx_n, ModelConstants.LANE_LINES_WIDTH), 0.2, dtype=np.float32),
    "lane_lines_prob": np.array([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]], dtype=np.float32),
    "road_edges": np.zeros((1, ModelConstants.NUM_ROAD_EDGES, idx_n, ModelConstants.ROAD_EDGES_WIDTH), dtype=np.float32),
    "road_edges_stds": np.full((1, ModelConstants.NUM_ROAD_EDGES, idx_n, ModelConstants.ROAD_EDGES_WIDTH), 0.3, dtype=np.float32),
    "lead": np.ones((1, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float32),
    "lead_stds": np.full((1, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), 0.4, dtype=np.float32),
    "lead_prob": np.array([[0.9, 0.5, 0.1]], dtype=np.float32),
    "desire_state": np.full((1, ModelConstants.DESIRE_PRED_WIDTH), 1.0 / ModelConstants.DESIRE_PRED_WIDTH, dtype=np.float32),
    "desire_pred": np.full((1, ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_PRED_WIDTH), 0.05, dtype=np.float32),
    "meta": np.zeros((1, 55), dtype=np.float32),
    "pose": np.array([[0.1, 0.2, 0.3, 0.01, 0.02, 0.03]], dtype=np.float32),
    "pose_stds": np.full((1, ModelConstants.POSE_WIDTH), 0.01, dtype=np.float32),
    "wide_from_device_euler": np.array([[0.01, 0.02, 0.03]], dtype=np.float32),
    "wide_from_device_euler_stds": np.full((1, ModelConstants.WIDE_FROM_DEVICE_WIDTH), 0.02, dtype=np.float32),
    "road_transform": np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    "road_transform_stds": np.full((1, ModelConstants.POSE_WIDTH), 0.03, dtype=np.float32),
  }

  data["meta"][0, Meta.ENGAGED] = 0.7
  data["meta"][0, Meta.BRAKE_DISENGAGE] = np.array([0.15, 0.10, 0.05, 0.05, 0.05], dtype=np.float32)
  data["meta"][0, Meta.GAS_DISENGAGE] = np.array([0.02, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
  data["meta"][0, Meta.STEER_OVERRIDE] = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
  data["meta"][0, Meta.HARD_BRAKE_3] = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
  data["meta"][0, Meta.HARD_BRAKE_4] = np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
  data["meta"][0, Meta.HARD_BRAKE_5] = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
  data["meta"][0, Meta.GAS_PRESS] = np.array([0.01] * 6, dtype=np.float32)
  data["meta"][0, Meta.BRAKE_PRESS] = np.array([0.01] * 6, dtype=np.float32)
  return data


def test_fill_xyzt_and_fill_xyvat_copy_expected_fields():
  xyzt = DummyBuilder()
  fill_xyzt(xyzt, [0.0, 1.0], np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]))
  assert xyzt.t == [0.0, 1.0]
  assert xyzt.x == [1.0, 2.0]
  assert xyzt.y == [3.0, 4.0]
  assert xyzt.z == [5.0, 6.0]

  xyvat = DummyBuilder()
  fill_xyvat(xyvat, [0.0], np.array([1.0]), np.array([2.0]), np.array([3.0]), np.array([4.0]))
  assert xyvat.t == [0.0]
  assert xyvat.x == [1.0]
  assert xyvat.y == [2.0]
  assert xyvat.v == [3.0]
  assert xyvat.a == [4.0]


def test_fill_xyz_poly_sets_degree_plus_one_coefficients():
  builder = DummyBuilder()
  x = np.array(ModelConstants.T_IDXS)
  y = x * 2.0
  z = x * 0.0

  fill_xyz_poly(builder, ModelConstants.POLY_PATH_DEGREE, x, y, z)

  expected_len = ModelConstants.POLY_PATH_DEGREE + 1
  assert len(builder.xCoefficients) == expected_len
  assert len(builder.yCoefficients) == expected_len
  assert len(builder.zCoefficients) == expected_len


def test_fill_lane_line_meta_uses_center_lane_indices():
  lane_lines = [
    SimpleNamespace(y=[0.0]),
    SimpleNamespace(y=[1.1]),
    SimpleNamespace(y=[2.2]),
    SimpleNamespace(y=[3.3]),
  ]
  probs = [0.1, 0.2, 0.3, 0.4]
  builder = DummyBuilder()

  fill_lane_line_meta(builder, lane_lines, probs)

  assert builder.leftY == 1.1
  assert builder.leftProb == 0.2
  assert builder.rightY == 2.2
  assert builder.rightProb == 0.3


def test_fill_pose_msg_sets_valid_and_pose_fields():
  msg = DummyBuilder()
  outs = _build_model_output_data()

  fill_pose_msg(msg, outs, vipc_frame_id=123, vipc_dropped_frames=0, timestamp_eof=777, live_calib_seen=True)
  assert msg.valid is True
  assert msg.cameraOdometry.frameId == 123
  assert msg.cameraOdometry.timestampEof == 777
  assert len(msg.cameraOdometry.trans) == 3
  assert len(msg.cameraOdometry.rot) == 3
  assert len(msg.cameraOdometry.wideFromDeviceEuler) == ModelConstants.WIDE_FROM_DEVICE_WIDTH

  fill_pose_msg(msg, outs, vipc_frame_id=124, vipc_dropped_frames=2, timestamp_eof=778, live_calib_seen=True)
  assert msg.valid is False


def test_fill_model_msg_populates_core_contract_fields():
  base_msg = DummyBuilder()
  extended_msg = DummyBuilder()
  outs = _build_model_output_data()
  state = PublishState()

  fill_model_msg(
    base_msg=base_msg,
    extended_msg=extended_msg,
    net_output_data=outs,
    v_ego=12.0,
    delay=0.1,
    publish_state=state,
    vipc_frame_id=40,
    vipc_frame_id_extra=41,
    frame_id=42,
    frame_drop=0.1,
    timestamp_eof=123456,
    model_execution_time=0.02,
    valid=True,
  )

  model_v2 = extended_msg.modelV2
  assert model_v2.frameId == 40
  assert model_v2.frameIdExtra == 41
  assert model_v2.frameAge == 2
  assert model_v2.timestampEof == 123456
  assert model_v2.frameDropPerc == 10.0
  assert model_v2.modelExecutionTime == pytest.approx(0.02, abs=1e-6)

  assert len(model_v2.laneLines) == ModelConstants.NUM_LANE_LINES
  assert len(model_v2.roadEdges) == ModelConstants.NUM_ROAD_EDGES
  assert len(model_v2.leadsV3) == ModelConstants.LEAD_MHP_SELECTION
  assert len(model_v2.laneLineProbs) == ModelConstants.NUM_LANE_LINES
  assert len(model_v2.meta.desirePrediction) == ModelConstants.DESIRE_PRED_LEN * ModelConstants.DESIRE_PRED_WIDTH

  assert base_msg.drivingModelData.action.desiredCurvature == pytest.approx(0.02, abs=1e-6)
  assert model_v2.action.desiredCurvature == pytest.approx(0.02, abs=1e-6)
  assert model_v2.confidence in (ConfidenceClass.green, ConfidenceClass.yellow, ConfidenceClass.red)

