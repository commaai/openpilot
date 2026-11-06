import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

LIFT_FULL_SPEED = 30.0  # m/s, lift ramps up from 0 to the full boost by this speed
LIFT_GATE = [0.95, 1.0]  # slowest planned speed / v_ego: no lift at the low end, full lift at the high end


def get_accel_boost(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 0.0
  elif personality==log.LongitudinalPersonality.standard:
    return 0.2
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.4
  else:
    raise NotImplementedError("Longitudinal personality not supported")


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.lift_filter = FirstOrderFilter(0.0, 0.3, dt)

  def reset(self):
    self.lift_filter.x = 0.0

  def update(self, sm, output_a_target_e2e):
    boost = get_accel_boost(sm['selfdriveState'].personality)
    v_ego = sm['carState'].vEgo
    plan_v = sm['modelV2'].velocity.x or [0.0]
    curve_scale = np.interp(abs(sm['controlsState'].curvature) * v_ego**2, [1.0, 2.5], [1.0, 0.0])  # fade out boost on curves

    gate = np.interp(min(plan_v) / max(v_ego, 1.0), LIFT_GATE, [0.0, 1.0])  # lift gated by if the model plans to slow more than 5% in the next 10 s
    lift = self.lift_filter.update(curve_scale * boost * min(v_ego / LIFT_FULL_SPEED, 1.0) * gate)

    gain = 1.0 + (boost * curve_scale if output_a_target_e2e > 0.0 else 0.0)  # boost gas only, braking is left to the model
    return output_a_target_e2e * gain + lift  # gain makes model's own acceleration stronger, lift increases the overall acceleration
