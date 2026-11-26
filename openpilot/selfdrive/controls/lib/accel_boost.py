import numpy as np
from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

PLAN_GATE = [-0.5, 1.0]  # m/s the plan runs ahead of holding the current speed: no boost at 0.5 behind, full boost at 1 ahead


def get_accel_boost(personality=log.LongitudinalPersonality.standard):
  # extra accel on top of the model's in m/s^2, and the extra fraction of the model's accel when speeding up
  if personality == log.LongitudinalPersonality.relaxed:
    return 0.0, 0.0
  elif personality == log.LongitudinalPersonality.standard:
    return 0.15, 0.1
  elif personality == log.LongitudinalPersonality.aggressive:
    return 0.3, 0.2
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_plan_ahead(position, v_ego: float) -> float:
  # m/s the plan's average speed over the next 10 s (distance / time) runs ahead of the current speed
  return float(position.x[-1] / position.t[-1] - max(v_ego, 1.0)) if len(position.x) else PLAN_GATE[0]


class AccelBoost:
  def __init__(self, dt: float = DT_MDL):
    self.plan_ahead_filter = FirstOrderFilter(0.0, 0.3, dt, initialized=False)  # smooths 1-3 frame blips in the plan
    self.bias_filter = FirstOrderFilter(0.0, 1.0, dt)  # drops at once when the plan slows and comes back over ~1 s

  def reset(self) -> None:
    self.plan_ahead_filter.initialized = False
    self.bias_filter.x = 0.0

  def update(self, sm, output_a_target_e2e: float, curve_offset: float, bias_scale: float = 1.0) -> float:
    plan_ahead = get_plan_ahead(sm['modelV2'].position, sm['carState'].vEgo)
    return output_a_target_e2e + self.boost(sm['selfdriveState'].personality, output_a_target_e2e, plan_ahead, curve_offset, bias_scale)

  def boost(self, personality, output_a_target_e2e: float, plan_ahead: float, curve_offset: float, bias_scale: float = 1.0) -> float:
    bias, gain = get_accel_boost(personality)
    self.plan_ahead_filter.update(plan_ahead)
    gate = float(np.interp(self.plan_ahead_filter.x, PLAN_GATE, [0.0, 1.0])) if curve_offset == 0.0 else 0.0
    target_bias = bias * bias_scale * gate
    self.bias_filter.x = min(self.bias_filter.x, target_bias)  # drop at once when a slowdown shows up in the plan
    self.bias_filter.update(target_bias)  # come back slowly once it clears

    gas_boost = gate * gain * max(output_a_target_e2e, 0.0)  # boost gas only while road is clear; braking is left to the model
    return gas_boost + self.bias_filter.x
