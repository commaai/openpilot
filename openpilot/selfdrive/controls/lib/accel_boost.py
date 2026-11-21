import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

LIFT_GATE_SPEED = [-0.5, 1.0]  # m/s the plan runs ahead of holding the current speed: no boost at 0.5 behind, full boost at 1 ahead
PLAN_AHEAD_TAU = 0.3  # s, smooths 1-3 frame blips in the plan before the gate
LIFT_REARM_TAU = 2.0  # s, the lift drops at once when the gate closes and comes back with this time constant
BRAKE_BOOST = 1.1  # braking gain at speed, faded in from 10 to 20 m/s


def get_accel_boost(personality=log.LongitudinalPersonality.standard):
  # extra accel on top of the model's in m/s^2, and the gain on the model's accel when speeding up
  if personality==log.LongitudinalPersonality.relaxed:
    return 0.0, 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 0.2, 1.2
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.4, 1.4
  else:
    raise NotImplementedError("Longitudinal personality not supported")


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.plan_ahead_filter = FirstOrderFilter(0.0, PLAN_AHEAD_TAU, dt, initialized=False)
    self.arm = 0.0
    self.total_boost = 0.0

  def reset(self):
    self.plan_ahead_filter.initialized = False
    self.arm = 0.0
    self.total_boost = 0.0

  def update(self, sm, output_a_target_e2e, curve_offset):
    lift, gas_gain = get_accel_boost(sm['selfdriveState'].personality)
    v_ego = max(sm['carState'].vEgo, 1.0)

    pos = sm['modelV2'].position
    t, x = np.array(pos.t), np.array(pos.x)
    # m/s the plan's average speed runs ahead of holding the current speed, at its worst point after 1 s
    plan_ahead = ((x - v_ego * t) / np.maximum(t, 1e-3))[t >= 1.0].min() if len(x) else LIFT_GATE_SPEED[0]
    plan_ahead = self.plan_ahead_filter.update(plan_ahead)
    gate = np.interp(plan_ahead, LIFT_GATE_SPEED, [0.0, 1.0]) if curve_offset == 0.0 else 0.0  # clear road ahead, none while slowing for a curve
    self.arm = min(gate, self.arm + (gate - self.arm) * self.dt / (LIFT_REARM_TAU + self.dt))  # falls at once, re-arms slowly

    if output_a_target_e2e > 0.0:
      gain = 1.0 + gate * (gas_gain - 1.0)  # boost gas only while the road ahead is clear
    elif gas_gain > 1.0:
      gain = np.interp(v_ego, [10.0, 20.0], [1.0, BRAKE_BOOST])  # brake a little harder at speed, relaxed is left to the model
    else:
      gain = 1.0
    self.total_boost = output_a_target_e2e * (gain - 1.0) + self.arm * lift
    return output_a_target_e2e + self.total_boost
