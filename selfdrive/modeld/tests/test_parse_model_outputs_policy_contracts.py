"""
``Parser.parse_policy_outputs`` contract tests including optional keys.

LOW-LEVEL §4.1 P1; optional branches match ``parse_model_outputs.py`` policy path.

Maps: R1.
"""

from __future__ import annotations

import numpy as np

from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.tests.modeld_parse_fixtures import minimal_policy_parse_outputs


def test_parse_policy_outputs_core_shapes():
  outs = minimal_policy_parse_outputs(batch=1, with_optional=False)
  Parser().parse_policy_outputs(outs)

  assert outs["plan"].shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
  assert outs["lane_lines"].shape == (
    1,
    ModelConstants.NUM_LANE_LINES,
    ModelConstants.IDX_N,
    ModelConstants.LANE_LINES_WIDTH,
  )
  assert outs["road_edges"].shape == (
    1,
    ModelConstants.NUM_ROAD_EDGES,
    ModelConstants.IDX_N,
    ModelConstants.ROAD_EDGES_WIDTH,
  )
  assert outs["sim_pose"].shape == (1, ModelConstants.POSE_WIDTH)
  assert outs["lead"].shape == (1, ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH)
  assert outs["desire_state"].shape == (1, ModelConstants.DESIRE_PRED_WIDTH)
  assert "lat_planner_solution" not in outs
  np.testing.assert_allclose(outs["desire_state"].sum(axis=-1), np.ones(1), atol=1e-5)


def test_parse_policy_outputs_optional_lat_planner_and_desired_curvature():
  outs = minimal_policy_parse_outputs(batch=1, with_optional=True)
  Parser().parse_policy_outputs(outs)

  assert outs["lat_planner_solution"].shape == (
    1,
    ModelConstants.IDX_N,
    ModelConstants.LAT_PLANNER_SOLUTION_WIDTH,
  )
  assert outs["desired_curvature"].shape == (1, ModelConstants.DESIRED_CURV_WIDTH)


def test_parse_policy_outputs_probabilities_sigmoid_bounded():
  outs = minimal_policy_parse_outputs(batch=1, with_optional=False)
  outs["lead_prob"][:] = np.array([[0.0, 2.0, -3.0]], dtype=np.float32)
  outs["lane_lines_prob"][:] = 1.0
  Parser().parse_policy_outputs(outs)

  assert np.all(outs["lead_prob"] > 0.0) and np.all(outs["lead_prob"] < 1.0)
  assert np.all(outs["lane_lines_prob"] > 0.0) and np.all(outs["lane_lines_prob"] < 1.0)


def test_parse_policy_outputs_plan_stds_positive_after_mdn():
  outs = minimal_policy_parse_outputs(batch=1, with_optional=False)
  Parser().parse_policy_outputs(outs)

  assert outs["plan_stds"].shape == outs["plan"].shape
  assert np.all(np.isfinite(outs["plan"]))
  assert np.all(np.isfinite(outs["plan_stds"]))
  assert np.all(outs["plan_stds"] > 0.0)
