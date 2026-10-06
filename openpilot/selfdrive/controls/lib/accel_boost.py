import numpy as np
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

ACCEL_BOOST_MAX = 0.2
ACCEL_BOOST_RATE = 0.025
ACCEL_BOOST_DECAY_RATE = 0.1
ACCEL_BOOST_PER_OVERRIDE = 0.05
ACCEL_BOOST_MIN_SPEED = 10 * CV.MPH_TO_MS


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.value = 0.0
    self.override_boost = 0.0
    self.model_limited = False

  def update(self, sm, output_a_target_e2e, output_a_target_mpc, a_cruise):
    enabled = sm['selfdriveState'].enabled
    gas_pressed = sm['carState'].gasPressed
    v_ego = sm['carState'].vEgo
    model_limited = (sm['selfdriveState'].experimentalMode and
                     output_a_target_e2e < min(output_a_target_mpc, a_cruise) - 0.1)

    if not enabled or not gas_pressed:
      self.override_boost = 0.0
      # Latch eligibility before the override changes the lead plan.
      self.model_limited = enabled and model_limited

    if not enabled:
      self.value = 0.0
    elif v_ego < ACCEL_BOOST_MIN_SPEED:
      self.value = max(0.0, self.value - ACCEL_BOOST_DECAY_RATE * self.dt)
    elif gas_pressed and self.model_limited:
      increase = min(ACCEL_BOOST_RATE * self.dt, ACCEL_BOOST_PER_OVERRIDE - self.override_boost, ACCEL_BOOST_MAX - self.value)
      self.value += increase
      self.override_boost += increase

  def apply(self, accel):
    return accel + np.interp(accel, [-1.0, -0.5, 5.0], [0.0, self.value, self.value], right=0.0)
