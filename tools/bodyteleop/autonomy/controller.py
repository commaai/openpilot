from __future__ import annotations

import math

from openpilot.tools.bodyteleop.autonomy.config import AutonomyConfig
from openpilot.tools.bodyteleop.autonomy.state_machine import BehaviorState


def _clip(x: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, x))


def compute_axes(state: BehaviorState,
                 target_bearing_deg: float,
                 target_distance_m: float,
                 cfg: AutonomyConfig) -> tuple[float, float]:
  if state != BehaviorState.ADVANCE:
    return 0.0, 0.0

  distance_error = max(0.0, target_distance_m - cfg.creep_target_distance_m)
  forward = _clip(distance_error / 2.0, 0.05, cfg.max_forward_axis)

  turn = _clip(math.radians(target_bearing_deg) / math.radians(45.0), -cfg.max_turn_axis, cfg.max_turn_axis)
  return forward, turn
