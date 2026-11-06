import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

LIFT_TARGET = 0.3  # m/s^2
LIFT_PER_SPEED = 0.01  # m/s^2 per m/s
LIFT_GATE = [0.95, 1.0]  # slowest planned speed / v_ego: no lift at the low end, full lift at the high end


def get_accel_boost(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 0.0
  elif personality==log.LongitudinalPersonality.standard:
    return 0.2
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.3
  else:
    raise NotImplementedError("Longitudinal personality not supported")


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.lift_filter = FirstOrderFilter(0.0, 0.3, dt)

  def reset(self):
    self.lift_filter.x = 0.0

  def update(self, sm, output_a_target_e2e):
    personality = sm['selfdriveState'].personality
    v_ego = sm['carState'].vEgo
    plan_v = sm['modelV2'].velocity.x
    curve_scale = np.interp(abs(sm['controlsState'].curvature) * v_ego**2, [1.0, 2.5], [1.0, 0.0])  # m/s^2 of lataccel, fade out boost on curves

    # Aggressive only: lift, unless the model plans to slow down more than 5% in the next 10 s
    lift_target = 0.0
    if personality == log.LongitudinalPersonality.aggressive and len(plan_v):
      gate = np.interp(min(plan_v) / max(v_ego, 1.0), LIFT_GATE, [0.0, 1.0])
      lift_target = curve_scale * min(LIFT_PER_SPEED * v_ego, LIFT_TARGET) * gate
    lift = self.lift_filter.update(lift_target)

    gain = 1.0 + get_accel_boost(personality) * curve_scale if output_a_target_e2e > 0.0 else 1.0  # gas only, braking is left to the model
    return (output_a_target_e2e * gain) + lift
