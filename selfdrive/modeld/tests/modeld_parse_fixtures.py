"""
Minimal ``net_output``-style dicts for ``Parser.parse_*`` tests only.

Keeps parse fixtures separate from ``modeld_test_fixtures`` (fill / DummyBuilder).

Maps: infra (R1 — deterministic parser inputs).
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants


def minimal_vision_parse_outputs(batch: int = 1) -> dict[str, np.ndarray]:
  """Keys required by ``Parser.parse_vision_outputs``."""
  pw = ModelConstants.POSE_WIDTH
  dlen, dw = ModelConstants.DESIRE_PRED_LEN, ModelConstants.DESIRE_PRED_WIDTH

  desire_pred = np.zeros((batch, dlen * dw), dtype=np.float32)
  meta = np.zeros((batch, 55), dtype=np.float32)

  return {
    "pose": np.zeros((batch, pw * 2), dtype=np.float32),
    "wide_from_device_euler": np.zeros((batch, ModelConstants.WIDE_FROM_DEVICE_WIDTH * 2), dtype=np.float32),
    "road_transform": np.zeros((batch, pw * 2), dtype=np.float32),
    "desire_pred": desire_pred,
    "meta": meta,
  }


def minimal_policy_parse_outputs(batch: int = 1, *, with_optional: bool = False) -> dict[str, np.ndarray]:
  """Keys required by ``Parser.parse_policy_outputs``; optional lat planner + desired curvature."""
  idx_n = ModelConstants.IDX_N
  plan_w = ModelConstants.PLAN_WIDTH
  n_plan = idx_n * plan_w
  plan_hyp = n_plan * 2 + ModelConstants.PLAN_MHP_SELECTION
  plan_raw = np.zeros((batch, ModelConstants.PLAN_MHP_N * plan_hyp), dtype=np.float32)

  n_ll = ModelConstants.NUM_LANE_LINES * idx_n * ModelConstants.LANE_LINES_WIDTH
  n_re = ModelConstants.NUM_ROAD_EDGES * idx_n * ModelConstants.ROAD_EDGES_WIDTH
  n_lead = ModelConstants.LEAD_TRAJ_LEN * ModelConstants.LEAD_WIDTH
  lead_hyp = n_lead * 2 + ModelConstants.LEAD_MHP_SELECTION

  outs: dict[str, np.ndarray] = {
    "plan": plan_raw,
    "lane_lines": np.zeros((batch, n_ll * 2), dtype=np.float32),
    "road_edges": np.zeros((batch, n_re * 2), dtype=np.float32),
    "sim_pose": np.zeros((batch, ModelConstants.POSE_WIDTH * 2), dtype=np.float32),
    "lead": np.zeros((batch, ModelConstants.LEAD_MHP_N * lead_hyp), dtype=np.float32),
    "lead_prob": np.zeros((batch, ModelConstants.LEAD_MHP_SELECTION), dtype=np.float32),
    "lane_lines_prob": np.zeros((batch, ModelConstants.NUM_LANE_LINES * 2), dtype=np.float32),
    "desire_state": np.zeros((batch, ModelConstants.DESIRE_PRED_WIDTH), dtype=np.float32),
  }

  if with_optional:
    n_lat = idx_n * ModelConstants.LAT_PLANNER_SOLUTION_WIDTH
    outs["lat_planner_solution"] = np.zeros((batch, n_lat * 2), dtype=np.float32)
    outs["desired_curvature"] = np.zeros((batch, ModelConstants.DESIRED_CURV_WIDTH * 2), dtype=np.float32)

  # Make plan non-degenerate so MDN branch runs (stds positive after safe_exp)
  plan_mu = plan_raw.reshape(batch, ModelConstants.PLAN_MHP_N, -1)[:, :, :n_plan]
  plan_mu[:] = 0.01 * np.random.default_rng(0).standard_normal(plan_mu.shape).astype(np.float32)

  return outs
