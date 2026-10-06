import numpy as np
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

ACCEL_BOOST_MAX = 0.2
ACCEL_BOOST_RATE = 0.025
ACCEL_BOOST_PER_OVERRIDE = 0.05
ACCEL_BOOST_MIN_SPEED = 10 * CV.MPH_TO_MS


def get_speed_scale(v_ego):
  return np.interp(v_ego, [ACCEL_BOOST_MIN_SPEED, 2 * ACCEL_BOOST_MIN_SPEED], [0.0, 1.0])


class AccelBoost:
  def __init__(self):
    self.total_boost = 0.0
    self.boost_this_override = 0.0
    self.boost_eligible = False

  def update(self, sm, output_a_target_e2e, output_a_target_mpc, a_cruise):
    enabled = sm['selfdriveState'].enabled
    gas_pressed = sm['carState'].gasPressed
    v_ego = sm['carState'].vEgo
    model_limited = (sm['selfdriveState'].experimentalMode and
                     self.apply(output_a_target_e2e, v_ego) < min(output_a_target_mpc, a_cruise) - 0.1)

    if not enabled or not gas_pressed:
      self.boost_this_override = 0.0
      # Latch eligibility before the override changes the lead plan.
      self.boost_eligible = enabled and model_limited

    if not enabled:
      self.total_boost = 0.0
    elif gas_pressed and self.boost_eligible:
      increase = min(ACCEL_BOOST_RATE * DT_MDL * get_speed_scale(v_ego),
                     ACCEL_BOOST_PER_OVERRIDE - self.boost_this_override, ACCEL_BOOST_MAX - self.total_boost)
      self.total_boost += increase
      self.boost_this_override += increase

  def apply(self, accel, v_ego):
    speed_scale = get_speed_scale(v_ego)
    return accel + speed_scale * np.interp(accel, [-1.0, -0.5, 5.0], [0.0, self.total_boost, self.total_boost], right=0.0)
