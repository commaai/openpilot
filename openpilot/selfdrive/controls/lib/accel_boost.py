import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

LIFT_GATE_SPEED = [-0.5, 1.0]  # m/s the plan runs ahead of holding the current speed: no boost at 0.5 behind, full boost at 1 ahead


def get_accel_boost(personality=log.LongitudinalPersonality.standard):
  # extra accel on top of the model's in m/s^2, and the extra fraction of the model's accel when speeding up
  if personality==log.LongitudinalPersonality.relaxed:
    return 0.0, 0.0
  elif personality==log.LongitudinalPersonality.standard:
    return 0.2, 0.1
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.4, 0.2
  else:
    raise NotImplementedError("Longitudinal personality not supported")


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.plan_ahead_filter = FirstOrderFilter(0.0, 0.3, dt, initialized=False)  # smooths 1-3 frame blips in the plan
    self.lift_filter = FirstOrderFilter(0.0, 2.0, dt)  # the lift drops at once when the plan slows and comes back over ~2 s

  def reset(self):
    self.plan_ahead_filter.initialized = False
    self.lift_filter.x = 0.0

  def update(self, sm, output_a_target_e2e, curve_offset):
    lift, gain = get_accel_boost(sm['selfdriveState'].personality)
    pos = sm['modelV2'].position
    # m/s the plan's average speed over the next 10 s (distance / time) runs ahead of the current speed
    plan_ahead = pos.x[-1] / pos.t[-1] - max(sm['carState'].vEgo, 1.0) if len(pos.x) else LIFT_GATE_SPEED[0]
    self.plan_ahead_filter.update(plan_ahead)
    gate = np.interp(self.plan_ahead_filter.x, LIFT_GATE_SPEED, [0.0, 1.0]) if curve_offset == 0.0 else 0.0  # clear road ahead, none while slowing for a curve
    self.lift_filter.x = min(self.lift_filter.x, lift * gate)  # drop at once when a slowdown shows up in the plan
    self.lift_filter.update(lift * gate)  # come back slowly once it clears

    gas_boost = gate * gain * max(output_a_target_e2e, 0.0)  # boost gas only while the road ahead is clear, braking is left to the model
    return output_a_target_e2e + gas_boost + self.lift_filter.x
