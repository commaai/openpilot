"""
Shared test doubles and minimal net-output dicts for ``modeld`` tests.

Use this for **new** tests only; keep upstream test modules unchanged.

Maps: infra (R1 support — consistent synthetic model outputs).
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld.constants import Meta, ModelConstants, Plan


class DummyBuilder:
  """Minimal capnp-like tree for exercising ``fill_model_msg`` without real capnp."""

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
    if count is None:
      value = DummyBuilder()
      self._attrs[name] = value
      return value
    values = [DummyBuilder() for _ in range(count)]
    self._attrs[name] = values
    return values


def minimal_net_output_data(batch: int = 1) -> dict[str, np.ndarray]:
  """Same tensor shapes as production ``fill_model_msg`` expects; values are benign constants."""
  if batch < 1:
    raise ValueError("batch must be >= 1")
  idx_n = ModelConstants.IDX_N

  plan = np.zeros((batch, idx_n, ModelConstants.PLAN_WIDTH), dtype=np.float32)
  plan_stds = np.full_like(plan, 0.1)
  for b in range(batch):
    for i, x in enumerate(ModelConstants.X_IDXS):
      plan[b, i, Plan.POSITION] = np.array([x, 0.1 * i, 0.0], dtype=np.float32)
      plan[b, i, Plan.VELOCITY] = np.array([10.0, 0.0, 0.0], dtype=np.float32)
      plan[b, i, Plan.ACCELERATION] = np.array([0.1, 0.0, 0.0], dtype=np.float32)
      plan[b, i, Plan.T_FROM_CURRENT_EULER] = np.array([0.0, 0.0, 0.01], dtype=np.float32)
      plan[b, i, Plan.ORIENTATION_RATE] = np.array([0.0, 0.0, 0.001], dtype=np.float32)

  data: dict[str, np.ndarray] = {
    "desired_curvature": np.full((batch, 1), 0.02, dtype=np.float32),
    "plan": plan,
    "plan_stds": plan_stds,
    "sim_pose": np.tile(np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]], dtype=np.float32), (batch, 1)),
    "sim_pose_stds": np.full((batch, ModelConstants.POSE_WIDTH), 0.05, dtype=np.float32),
    "lane_lines": np.zeros((batch, ModelConstants.NUM_LANE_LINES, idx_n, ModelConstants.LANE_LINES_WIDTH), dtype=np.float32),
    "lane_lines_stds": np.full((batch, ModelConstants.NUM_LANE_LINES, idx_n, ModelConstants.LANE_LINES_WIDTH), 0.2, dtype=np.float32),
    "lane_lines_prob": np.tile(
      np.array([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]], dtype=np.float32),
      (batch, 1),
    ),
    "road_edges": np.zeros(
      (batch, ModelConstants.NUM_ROAD_EDGES, idx_n, ModelConstants.ROAD_EDGES_WIDTH),
      dtype=np.float32,
    ),
    "road_edges_stds": np.full(
      (batch, ModelConstants.NUM_ROAD_EDGES, idx_n, ModelConstants.ROAD_EDGES_WIDTH),
      0.3,
      dtype=np.float32,
    ),
    "lead": np.ones((batch, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float32),
    "lead_stds": np.full((batch, 3, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), 0.4, dtype=np.float32),
    "lead_prob": np.tile(np.array([[0.9, 0.5, 0.1]], dtype=np.float32), (batch, 1)),
    "desire_state": np.full((batch, ModelConstants.DESIRE_PRED_WIDTH), 1.0 / ModelConstants.DESIRE_PRED_WIDTH, dtype=np.float32),
    "desire_pred": np.full((batch, ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_PRED_WIDTH), 0.05, dtype=np.float32),
    "meta": np.zeros((batch, 55), dtype=np.float32),
    "pose": np.tile(np.array([[0.1, 0.2, 0.3, 0.01, 0.02, 0.03]], dtype=np.float32), (batch, 1)),
    "pose_stds": np.full((batch, ModelConstants.POSE_WIDTH), 0.01, dtype=np.float32),
    "wide_from_device_euler": np.tile(np.array([[0.01, 0.02, 0.03]], dtype=np.float32), (batch, 1)),
    "wide_from_device_euler_stds": np.full((batch, ModelConstants.WIDE_FROM_DEVICE_WIDTH), 0.02, dtype=np.float32),
    "road_transform": np.tile(np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float32), (batch, 1)),
    "road_transform_stds": np.full((batch, ModelConstants.POSE_WIDTH), 0.03, dtype=np.float32),
  }

  for b in range(batch):
    data["meta"][b, Meta.ENGAGED] = 0.7
    data["meta"][b, Meta.BRAKE_DISENGAGE] = np.array([0.15, 0.10, 0.05, 0.05, 0.05], dtype=np.float32)
    data["meta"][b, Meta.GAS_DISENGAGE] = np.array([0.02, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
    data["meta"][b, Meta.STEER_OVERRIDE] = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
    data["meta"][b, Meta.HARD_BRAKE_3] = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    data["meta"][b, Meta.HARD_BRAKE_4] = np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    data["meta"][b, Meta.HARD_BRAKE_5] = np.array([0.01, 0.01, 0.01, 0.01, 0.01], dtype=np.float32)
    data["meta"][b, Meta.GAS_PRESS] = np.array([0.01] * 6, dtype=np.float32)
    data["meta"][b, Meta.BRAKE_PRESS] = np.array([0.01] * 6, dtype=np.float32)

  return data
