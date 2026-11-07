import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

LIFT_GATE_STDS = [-1.0, 2.0]  # less lift if the future plan is behind compared to maintaining current speed, more if ahead
BRAKE_BOOST = 1.1  # braking gain at speed, faded in from 10 to 20 m/s


def get_accel_boost(personality=log.LongitudinalPersonality.standard):
  # extra accel on top of the model's, in m/s^2
  if personality==log.LongitudinalPersonality.relaxed:
    return 0.0
  elif personality==log.LongitudinalPersonality.standard:
    return 0.1
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.2
  else:
    raise NotImplementedError("Longitudinal personality not supported")


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.lift_filter = FirstOrderFilter(0.0, 0.3, dt)
    self.total_boost = 0.0

  def reset(self):
    self.lift_filter.x = 0.0
    self.total_boost = 0.0

  def update(self, sm, output_a_target_e2e):
    boost = get_accel_boost(sm['selfdriveState'].personality)
    v_ego = max(sm['carState'].vEgo, 1.0)
    curve_scale = np.interp(abs(sm['controlsState'].curvature) * v_ego**2, [1.0, 2.5], [1.0, 0.0])  # fade out boost on curves

    pos = sm['modelV2'].position
    t, x, x_std = np.array(pos.t), np.array(pos.x), np.array(pos.xStd)
    # how many stds the plan is ahead compared to holding the current speed, at its worst point after 1 s
    plan_ahead = ((x - v_ego * t) / x_std)[t >= 1.0].min() if len(x_std) else LIFT_GATE_STDS[0]
    gate = np.interp(plan_ahead, LIFT_GATE_STDS, [0.0, 1.0])
    lift = self.lift_filter.update(curve_scale * boost * gate)

    if output_a_target_e2e < 0.0 and boost > 0.0:
      gain = np.interp(v_ego, [10.0, 20.0], [1.0, BRAKE_BOOST])  # brake a little harder at speed, relaxed is left to the model
    else:
      gain = 1.0
    self.total_boost = output_a_target_e2e * (gain - 1.0) + lift
    return output_a_target_e2e + self.total_boost
